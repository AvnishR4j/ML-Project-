"""Generate SYNTHETIC labelled Hinglish messages (romanized Hindi + English) for spam-model augmentation.

These are template-generated, NOT real messages. They are used only as extra TRAINING data and are never
used for any test number. Output: data/synthetic/hinglish_spam_synthetic.csv  (text,label,source=synthetic)
   python scripts/make_synthetic_hinglish.py
"""
import random, pandas as pd
from pathlib import Path

R = random.Random(7)
pick = lambda xs: R.choice(xs)
bank = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Paytm", "PhonePe", "Bank of Baroda", "Kotak"]
amt = lambda: pick(["5,000", "10,000", "25,000", "50,000", "1,00,000", "2 lakh", "5 lakh", "499", "999", "1500"])
link = lambda: pick(["http://bit.ly/x%d" % R.randint(100, 999), "https://secure-%s.in/verify" % pick(["kyc", "bank", "upi", "claim"]), "www.%s-offer.com/win" % pick(["jio", "amazon", "flipkart", "gift"])])
ph = lambda: "9%09d" % R.randint(0, 999999999)
urg = ["abhi", "turant", "jaldi se", "aaj hi", "24 ghante ke andar", "kal tak", "immediately"]

SPAM = [
 lambda: f"Dear customer aapka {pick(bank)} account {pick(urg)} block ho jayega. KYC update karne ke liye click karein {link()}",
 lambda: f"{pick(bank)}: aapka KYC expire ho gaya hai. {pick(urg)} update na kiya toh account band ho jayega. Link: {link()}",
 lambda: f"Badhai ho! Aapne Rs {amt()} ka lucky draw jeeta hai. Paisa paane ke liye {pick(urg)} {ph()} par call karein",
 lambda: f"Congratulations aap {pick(['iPhone 15','Maruti car','Rs '+amt()])} jeet gaye hain! Claim karne ke liye {link()} par click karo",
 lambda: f"Aapka electricity bill pending hai, aaj raat 9.30 baje connection kat jayega. Turant is number par sampark karein {ph()}",
 lambda: f"Aapka SIM {pick(urg)} band hoga. Verification ke liye apna Aadhaar number aur OTP bhejein {ph()}",
 lambda: f"Ghar baithe kamayein Rs {amt()} roz! Sirf mobile chahiye. WhatsApp karein {ph()} abhi join karo, seats limited hain",
 lambda: f"Part time job offer: YouTube videos like karke Rs {amt()} kamao. Registration free, Telegram par join karein {link()}",
 lambda: f"Aapka loan Rs {amt()} approve ho gaya hai bina documents ke. Abhi apply karein {link()} sirf aaj ke liye",
 lambda: f"Pre-approved personal loan {amt()} 0% interest. Processing fee Rs {amt()} bhejein aur paisa turant account mein paayein {ph()}",
 lambda: f"Aapka parcel customs mein ruka hai. Delivery ke liye Rs {amt()} fee pay karein {link()} warna wapas bhej diya jayega",
 lambda: f"Aapke account se Rs {amt()} debit hue hain. Agar aapne nahi kiya toh {pick(urg)} refund ke liye {link()} par jaayein",
 lambda: f"UPI refund pending: Rs {amt()} aapko milna baaki hai. PIN daal kar {link()} par claim karein",
 lambda: f"Bitcoin mein invest karo aur 7 din mein paisa double karo! Guaranteed profit. Join karein {link()} {pick(urg)}",
 lambda: f"Share market tips: sure shot profit Rs {amt()} har din. Premium group join karein sirf Rs {amt()} mein {ph()}",
 lambda: f"URGENT: aapka {pick(bank)} debit card block hai. Card number aur CVV bhejkar unblock karein {ph()}",
 lambda: f"Free recharge Rs {amt()} paane ke liye {link()} par click karein aur apne dosto ko share karein",
 lambda: f"Aapka PAN card update nahi hua hai, penalty lagegi. {pick(urg)} {link()} par update karein",
 lambda: f"Sarkari yojana: har parivar ko Rs {amt()} milenge. Form bharne ke liye {link()} abhi register karein",
 lambda: f"Hello sir, main {pick(bank)} se bol raha hun. Aapka card upgrade hoga, bas OTP bata dijiye jo abhi aapke phone par aaya hai",
 lambda: f"Dating app: sundar ladkiyan aapse baat karna chahti hain. Free join karein {link()} 18+ only",
 lambda: f"Aapko {pick(['Amazon','Flipkart','Jio'])} ki taraf se Rs {amt()} ka gift voucher mila hai. Sirf {pick(urg)} claim karein {link()}",
 lambda: f"WhatsApp gold subscription expire ho raha hai, free renew karne ke liye {link()} click karein",
]

SUBJ = ["kal", "aaj", "parso", "shaam ko", "weekend par", "subah"]
NAMES = ["Rahul", "Priya", "Amit", "Neha", "Sandeep", "Pooja", "Vikas", "Anjali", "bhaiya", "didi", "yaar"]
HAM = [
 lambda: f"{pick(NAMES)}, {pick(SUBJ)} milte hain kya? Canteen mein chai peeni hai",
 lambda: f"Bhai assignment ka deadline {pick(SUBJ)} hai, tune submit kar diya?",
 lambda: f"Mummy ne kaha hai ki {pick(SUBJ)} ghar jaldi aa jaana, dinner mein paneer bana hai",
 lambda: f"Main {pick(SUBJ)} late ho jaunga, meeting lambi chal rahi hai. Tum log start kar lo",
 lambda: f"Aaj ka match dekha? Kohli ne kya shot maara yaar, maza aa gaya",
 lambda: f"Tera notes bhej dena please, kal ki class miss ho gayi thi",
 lambda: f"Happy birthday {pick(NAMES)}! Party kab de raha hai? Treat toh banti hai",
 lambda: f"Bhai paise {pick(SUBJ)} wapas kar dunga, abhi thoda tight chal raha hai",
 lambda: f"Order {pick(['Zomato','Swiggy','Amazon'])} se aa gaya hai, delivery wale bhaiya neeche khade hain",
 lambda: f"Chai thandi ho rahi hai, jaldi aa ja. Sab log wait kar rahe hain",
 lambda: f"{pick(NAMES)} ka phone nahi lag raha, kisi ke paas uska dusra number hai?",
 lambda: f"Film ka review kaisa hai? Dekhne jaana chahiye ya paisa barbad hoga",
 lambda: f"Kal exam hai aur maine kuch bhi nahi padha, ab kya karun",
 lambda: f"Reminder: {pick(SUBJ)} 5 baje dentist ki appointment hai, time par pahunch jaana",
 lambda: f"Wifi ka password kya hai? Mera net bilkul slow chal raha hai",
 lambda: f"Bahut mast gana hai ye, baar baar sunne ka mann karta hai",
 lambda: f"Tum log trip ka plan final karo, main {pick(['Goa','Manali','Jaipur'])} chalne ko ready hun",
 lambda: f"Sorry yaar, kal tumhara call nahi utha paya. Kaam mein busy tha, ab free hun",
 lambda: f"Aaj office mein bahut kaam tha, thak gaya hun. Khana kha liya tune?",
 lambda: f"Bhai Rs {amt()} ka mobile liya hai kal, camera ekdum zabardast hai",
 lambda: f"Bank ki line mein khada hun {pick(SUBJ)} se, pura time kharab ho gaya",
 lambda: f"Dadi ki tabiyat theek hai ab, doctor ne kaha hai ki ghabrane ki baat nahi",
]


def make(n_spam=700, n_ham=700):
    rows, seen = [], set()
    for gens, lab, n in ((SPAM, "spam", n_spam), (HAM, "ham", n_ham)):
        tries = 0
        while sum(1 for r in rows if r[1] == lab) < n and tries < n * 50:
            tries += 1
            t = pick(gens)()
            if R.random() < 0.3: t = t.lower()
            if t not in seen: seen.add(t); rows.append((t, lab, "synthetic"))
    return pd.DataFrame(rows, columns=["text", "label", "source"])


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "hinglish_spam_synthetic.csv"
    d = make(); d.to_csv(out, index=False); print(d.label.value_counts().to_dict(), "->", out)
