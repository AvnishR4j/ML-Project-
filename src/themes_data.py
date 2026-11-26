"""Corpora for theme discovery, with topic labels that the clustering never sees (used only to MEASURE it).

English  (topic label | source)
  banking_cards      CLINC150 'banking' + 'credit_cards' domains, BANKING77
  kitchen_dining, travel, auto_commute, work_pay, utility_time, home_assistant, small_talk, meta_assistant   CLINC150 domains
  product_reviews    Hu & Liu customer reviews (camera, phone, mp3 player, router, software, ...)
  movie_reviews      Rotten Tomatoes sentences
  tweets             TweetEval tweets (open-topic social media)
  sms_chat           UCI SMS (genuine messages)
Hinglish
  Hinglish-TOP human-written queries (8 domains: alarm, event, messaging, music, navigation, reminder, timer, weather)
  IIIT-H YouTube comments on Salman Khan films  ->  'film_fan_comments'
  L3Cube-HingLID Hinglish tweets (open topics)  ->  'hinglish_tweets'
  author-written everyday messages (10 topics x 54; 60 % used for training, 40 % held out)  -> data/hinglish_topics_seed.csv
Out-of-topic test sets: CLINC150 'oos' queries (English), author-written Hinglish messages on 8 topics the model never sees.
"""
import json, re
import numpy as np, pandas as pd
from data import ROOT, RAW, load_spam, _read_hinglid
from data2 import clean_social, load_hinglish_sentiment

D = ROOT / "data"
CLINC_DOMAINS = {
    "banking_cards": "transfer transactions balance freeze_account pay_bill bill_balance bill_due interest_rate routing min_payment order_checks pin_change report_fraud account_blocked spending_history "
                     "credit_score report_lost_card credit_limit rewards_balance new_card application_status card_declined international_fees apr redeem_rewards credit_limit_change damaged_card replacement_card_duration improve_credit_score expiration_date",
    "kitchen_dining": "recipe restaurant_reviews calories nutrition_info restaurant_suggestion ingredients_list ingredient_substitution cook_time food_last meal_suggestion restaurant_reservation confirm_reservation how_busy cancel_reservation accept_reservations",
    "home_assistant": "shopping_list shopping_list_update next_song play_music update_playlist todo_list todo_list_update calendar calendar_update what_song order order_status reminder reminder_update smart_home",
    "auto_commute": "traffic directions gas gas_type distance current_location mpg oil_change_when oil_change_how jump_start uber schedule_maintenance last_maintenance tire_pressure tire_change",
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert travel_notification carry_on timezone vaccines translate flight_status international_visa lost_luggage plug_type exchange_rate",
    "utility_time": "time alarm share_location find_phone weather text spelling make_call timer date calculator measurement_conversion flip_coin roll_dice definition",
    "work_pay": "direct_deposit pto_request taxes payday w2 pto_balance pto_request_status next_holiday insurance insurance_change schedule_meeting pto_used meeting_schedule rollover_401k income",
    "small_talk": "greeting goodbye tell_joke where_are_you_from how_old_are_you what_is_your_name who_made_you thank_you what_can_i_ask_you what_are_your_hobbies do_you_have_pets are_you_a_bot meaning_of_life who_do_you_work_for fun_fact",
    "meta_assistant": "change_ai_name change_user_name cancel user_name reset_settings whisper_mode repeat no yes maybe change_language change_accent change_volume change_speed sync_device",
}
INTENT2DOMAIN = {i: d for d, s in CLINC_DOMAINS.items() for i in s.split()}


def _clinc():
    j = json.loads((RAW / "clinc150_full.json").read_text())
    out = {}
    for sp in ("train", "val", "test", "oos_test", "oos_val", "oos_train"):
        rows = []
        for t, i in j[sp]:
            if i == "oos":
                rows.append((t, "oos"))
            else:
                rows.append((t, INTENT2DOMAIN[i]))
        out[sp] = pd.DataFrame(rows, columns=["text", "topic"])
    return out


def _banking():
    tr = pd.read_csv(RAW / "banking77_train.csv"); te = pd.read_csv(RAW / "banking77_test.csv")
    f = lambda d: pd.DataFrame({"text": d.text, "topic": "banking_cards"})
    return f(tr), f(te)


def _product_reviews():
    rows = []
    for sub in ("product_reviews_1", "product_reviews_2"):
        for f in sorted((RAW / sub).glob("*.txt")):
            if f.name.lower().startswith("readme"):
                continue
            for line in f.read_text(errors="ignore").splitlines():
                if "##" in line and not line.startswith("*"):
                    s = line.split("##", 1)[1].strip()
                    if len(s.split()) >= 4:
                        rows.append(s)
    return pd.DataFrame({"text": rows, "topic": "product_reviews"})


def _movie():
    rows = []
    for f in ("rt-polarity.pos", "rt-polarity.neg"):
        rows += [l.strip() for l in (RAW / "sentence_polarity" / f).read_text(errors="ignore").splitlines() if len(l.split()) >= 4]
    return pd.DataFrame({"text": rows, "topic": "movie_reviews"})


def _tweets():
    t = (RAW / "tweeteval" / "train_text.txt").read_text(errors="ignore").splitlines()
    t = [clean_social(x.strip().strip('"')) for x in t]
    return pd.DataFrame({"text": [x for x in t if len(x.split()) >= 4], "topic": "tweets"})


def _sms():
    tr, te, _ = load_spam(); d = pd.concat([tr, te], ignore_index=True)
    return pd.DataFrame({"text": d[d.label == "ham"].text.values, "topic": "sms_chat"})


def _split(df, rng, cap_train, cap_eval, eval_frac=0.25):
    idx = rng.permutation(len(df)); n_eval = min(cap_eval, int(len(df) * eval_frac)); ev = df.iloc[idx[:n_eval]]; tr = df.iloc[idx[n_eval:n_eval + cap_train]]
    return tr, ev


def english(seed=42, cap_train=1500, cap_eval=400):
    """returns (train_pool, eval_set, oos_set); each a DataFrame(text, topic). Pools are capped per topic so no source dominates."""
    rng = np.random.default_rng(seed); c = _clinc(); btr, bte = _banking()
    tr_parts, ev_parts = [], []
    # CLINC and BANKING77 come with official train / test splits
    for d, n_tr, n_ev in ((pd.concat([c["train"], c["val"], btr]), cap_train, 0), (pd.concat([c["test"], bte]), 0, cap_eval)):
        for t, g in d.groupby("topic"):
            g = g.sample(frac=1, random_state=seed)
            if n_tr: tr_parts.append(g.iloc[:n_tr])
            if n_ev: ev_parts.append(g.iloc[:n_ev])
    for df in (_product_reviews(), _movie(), _tweets(), _sms()):
        a, b = _split(df, rng, cap_train, cap_eval); tr_parts.append(a); ev_parts.append(b)
    oos = pd.concat([c["oos_test"], c["oos_val"], c["oos_train"]], ignore_index=True)
    return pd.concat(tr_parts, ignore_index=True), pd.concat(ev_parts, ignore_index=True), oos


def _top(sp):
    d = pd.read_csv(RAW / f"hinglish_top_{sp}.tsv", sep="\t", quoting=3, on_bad_lines="skip")
    d = d.dropna(subset=["cs_query", "domain"]); d = d[d.domain.isin(["alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"])]
    return pd.DataFrame({"text": d.cs_query.astype(str).values, "topic": "top_" + d.domain.values})


def hinglish(seed=42, cap_train=1200, cap_eval=300):
    """returns (train_pool, eval_set, oos_set, extra_w2v_sentences)."""
    rng = np.random.default_rng(seed)
    top = pd.concat([_top("train"), _top("validation")]); top_te = _top("test")
    h_tr, h_te = load_hinglish_sentiment(); fan = pd.concat([h_tr, h_te], ignore_index=True)
    fan = pd.DataFrame({"text": fan.text.values, "topic": "film_fan_comments"})
    seed_df = pd.read_csv(D / "hinglish_topics_seed.csv")
    tr_parts, ev_parts = [], []
    for t, g in top.groupby("topic"): tr_parts.append(g.sample(frac=1, random_state=seed).iloc[:cap_train])
    for t, g in top_te.groupby("topic"): ev_parts.append(g.sample(frac=1, random_state=seed).iloc[:cap_eval])
    a, b = _split(fan, rng, cap_train, cap_eval); tr_parts.append(a); ev_parts.append(b)
    tr_parts.append(seed_df[seed_df.split == "train"][["text", "topic"]]); ev_parts.append(seed_df[seed_df.split == "eval"][["text", "topic"]])
    ood = pd.read_csv(D / "hinglish_topics_ood.csv")
    sents = _read_hinglid(RAW / "hinglid" / "train.txt"); idx = rng.choice(len(sents), min(60000, len(sents)), replace=False)
    tw = pd.DataFrame({"text": [" ".join(sents[i]) for i in idx if len(sents[i]) >= 5], "topic": "hinglish_tweets"})      # open-topic Hinglish tweets (L3Cube-HingLID)
    a, b = _split(tw, rng, cap_train, cap_eval); tr_parts.append(a); ev_parts.append(b)
    return pd.concat(tr_parts, ignore_index=True), pd.concat(ev_parts, ignore_index=True), ood, [" ".join(sents[i]) for i in idx]


if __name__ == "__main__":
    tr, ev, oos = english(); print("EN train", len(tr), "eval", len(ev), "oos", len(oos)); print(pd.concat([tr.topic.value_counts().rename("train"), ev.topic.value_counts().rename("eval")], axis=1))
    tr, ev, ood, w = hinglish(); print("HI train", len(tr), "eval", len(ev), "ood", len(ood), "w2v extra", len(w)); print(pd.concat([tr.topic.value_counts().rename("train"), ev.topic.value_counts().rename("eval")], axis=1))
    missing = [i for i in json.loads((RAW / "clinc150_full.json").read_text())["train"] if i[1] != "oos" and i[1] not in INTENT2DOMAIN]
    print("CLINC intents not mapped:", len(missing))
