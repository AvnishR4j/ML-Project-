"""Round-3 generated training messages (training only): conversational Hinglish scams ('Bhai/Yaar/Bro ...' openers), brand-impersonation alerts,
emergency money requests, job-form fees, penalty/notice scams, prize + shipping-fee scams, and heavily misspelled English prize/loan/reward spam.
Each scam type has a genuine twin (same people, brands and words, no money/OTP demand) so the model has to learn the ask, not the vocabulary."""
import random
OPEN = ["Bhai", "Yaar", "Bro", "Dost", "Boss", "Sir", "", "", "Dear customer,", "Hello,"]
BRAND = ["HDFC", "SBI", "ICICI", "Axis Bank", "Bank of Baroda", "PNB", "Paytm", "PhonePe", "Google Pay", "Apple ID", "iCloud", "Google account", "WhatsApp", "Instagram", "Facebook", "Amazon", "Flipkart", "Zomato", "Jio", "Airtel", "Netflix", "Microsoft"]
GOV = ["GST department", "Income Tax department", "election commission", "passport office", "RTO", "transport department", "PAN seva", "Aadhaar centre", "traffic police", "electricity board", "municipal corporation"]
SM = ["200", "299", "350", "500", "999", "1500", "2500", "5000"]; BG = ["5000", "7000", "10000", "12000", "20000", "50000", "1 lakh", "2 lakh"]
DAYS = ["2 din", "24 ghante", "3 din", "7 din", "48 hours", "aaj raat"]; WHY = ["hospital mein hoon", "accident ho gaya", "phone kho gaya", "wallet gum ho gaya", "station pe fasa hoon", "bahut mushkil mein hoon", "emergency hai"]
LK = lambda r: r.choice(["", "", " bit.ly/" + "".join(r.choice("abcdefgh12345") for _ in range(6)), " tinyurl.com/" + "".join(r.choice("abcdefgh12345") for _ in range(5)), " http://secure-" + "".join(r.choice("abcdefgh") for _ in range(5)) + ".xyz"])
PH = lambda r: str(r.choice([98, 97, 99, 70, 80])) + "".join(str(r.randrange(10)) for _ in range(8))
def pick(r, L): return r.choice(L)

def scam(r):
    o, b, g, sm, bg, dy, why, lk = pick(r, OPEN), pick(r, BRAND), pick(r, GOV), pick(r, SM), pick(r, BG), pick(r, DAYS), pick(r, WHY), LK(r)
    T = [f"{b} ka message - FD mature ho gayi, {bg} release karne ko {sm} fee bhar{lk}",
         f"{b} alert - tera account {dy} mein band ho jayega KYC ke bina, abhi update kar{lk}",
         f"{b} se call aaya tha, bol rahe the account block hoga, OTP share kar do verify karne ke liye",
         f"{b} security alert: aapka account hack hua hai, OTP share karein cancel karne ke liye{lk}",
         f"{b} storage full hai, data delete hone se pehle backup kar, link kholo{lk}",
         f"Tere phone mein dangerous virus hai, data safe karne ke liye ye app install kar{lk}",
         f"Main {why}, please {sm} transfer kar do is number pe {PH(r)}, aaj raat pakka return karunga",
         f"{why}, {sm} UPI kar do abhi {PH(r)} pe, ghar pahunchke wapas kar dunga pakka",
         f"Urgent paise chahiye, {why}. Is UPI id pe {sm} bhejo, kisi ko mat batana",
         f"Teacher chahiye {b} mein, monthly {r.choice(['30k', '40k', '55k'])} - interview ke liye {sm} ka form bharo{lk}",
         f"Job offer pakka hai, joining letter ke liye {sm} registration fee bhejo {PH(r)}",
         f"{g} ne penalty lagai hai {bg} ki, {dy} mein pay karo warna case hoga{lk}",
         f"{g} ka notice - documents upload karo warna application reject hogi{lk}",
         f"{g} notice: voter ID / PAN update karna zaroori hai, OTP ke saath verify karein{lk}",
         f"{b} sale mein aap winner hain - iPhone jeet liya! Delivery ke liye {sm} shipping bhar{lk}",
         f"{b} anniversary pe {sm} ka voucher mila hai, {dy} mein expire hoga, abhi claim karo{lk}",
         f"{b} ne tujhe select kiya hai - surprise gift jeet liya! Registration fee {sm} bhar{lk}",
         f"You won a {pick(r, ['iPhone 15', 'MacBook', 'Rs 5 lakh cash', 'car', 'gold coin'])} from {b}! Pay {sm} delivery charge to claim{lk}",
         f"Claim ur reward now!! limited time offer, dont miss out{lk}",
         f"Congratulations you have won a lottery of {pick(r, ['5 million', '1 million dollars', 'Rs 25 lakh'])}, reply with bank details{lk}",
         f"Get cheap loans with no credit check, approved in 5 minutes{lk}",
         f"Cheap loan approved, no documents needed, pay {sm} insurance fee first{lk}",
         f"Earn {pick(r, ['3000', '5000', '10000'])} daily from phone, join now{lk} pay {sm} registration",
         f"Your {b} reward points will expire today, redeem now{lk}"]
    t = r.choice(T); t = (o + " " + t).strip() if o and not o.endswith(",") else (o + " " + t).strip()
    return t

def genuine(r):
    o, b, g, sm, bg, dy, lk = pick(r, OPEN[:6]), pick(r, BRAND), pick(r, GOV), pick(r, SM), pick(r, BG), pick(r, DAYS), ""
    T = [f"{b} ka message aaya tha - FD mature ho gayi, branch jaake renew kara lena",
         f"{b} se message aaya, statement mail pe bhej di hai, dekh lena",
         f"{b} ka password reset kar diya maine, new password mummy ko bata diya",
         f"Main hospital mein hoon, mummy theek hain, shaam ko aa jana",
         f"Phone kho gaya tha lekin mil gaya, tension mat le",
         f"Wallet mil gaya station pe, kal de dunga tujhe",
         f"Yaar {sm} UPI kar diye the kal, mil gaye kya? Check kar lena",
         f"Tere {sm} wapas kar diye hain UPI se, thanks bhai",
         f"Bhai {b} se order aaya hai, delivery kal tak hogi",
         f"{g} jaana hai kal, saare documents ready hain, saath chalega?",
         f"GST return bhar diya hai, receipt mail pe aa gayi",
         f"{g} ke office mein form jama kar diya, token number mila hai",
         f"Teacher ne bola kal test hai, notes bhej dena",
         f"Interview kal {r.randrange(10, 17)} baje hai, formal kapde pehen ke aana",
         f"Zomato se {sm} ka discount mila hai, tu bhi app mein use kar le",
         f"Salary aa gayi, {sm} ghar bhej diye hain",
         f"Phone mein update aaya hai, wifi pe install kar lena",
         f"Is hafte {b} wali meeting reschedule ho gayi hai, mail dekh lena",
         f"Congratulations bhai, selection ho gaya! Party kab de raha hai",
         f"Congratulations, you have been selected for the college cricket team. Practice at 5 pm",
         f"Your loan application documents were received. Our branch will contact you in working hours",
         f"Hi, your {b} reward points statement is available in the official app",
         f"I will pay you back the {sm} tomorrow, thanks for helping",
         f"Did you win the prize at the college fest? Everyone is talking about it"]
    t = r.choice(T); return (o + " " + t).strip() if o and r.random() < 0.5 else t

def typo(t, r, p):
    if r.random() > p: return t
    out = []
    for w in t.split(" "):
        x = w
        if len(w) >= 5 and r.random() < 0.35:
            k = r.randrange(1, len(w) - 1); x = w[:k] + w[k + 1:]
        elif len(w) >= 4 and r.random() < 0.15 and any(c * 2 in w for c in "lnmstrp"):
            for c in "lnmstrp":
                if c * 2 in w: x = w.replace(c * 2, c, 1); break
        out.append(x)
    return " ".join(out)

def make3(n_each=1500, seed=31):
    r = random.Random(seed); out = [(typo(genuine(r), r, 0.25), "ham") for _ in range(n_each)] + [(typo(scam(r), r, 0.45), "spam") for _ in range(n_each)]
    seen = set(); res = []
    for t, l in out:
        if t not in seen: seen.add(t); res.append((t, l))
    return res
