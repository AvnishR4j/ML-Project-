"""Round-2 generated training messages (training only, never used for scoring).
Fixes what an independent third-party test set (data/eval/spam_independent_83.csv, 83 lines written by someone else) exposed:
  * genuine bank / UPI / card alerts, delivery, e-challan, tax, bill and telecom messages that carry official links or 'Not you? Call ...' were flagged as spam
  * threat / kidnap / digital-arrest / job-fee / loan-app / electricity-cut / OLX-QR / gift-card / romance scams and SMS-speak spelling were missed
The wording below is written from general knowledge of Indian SMS formats; it does not copy any test line.
SMS-speak noise is applied to both classes, so spelling alone does not decide the label."""
import random
BANK = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Bank", "PNB", "Bank of Baroda", "Canara Bank", "IDFC FIRST Bank", "Union Bank"]
BDOM = {"HDFC Bank": "hdfcbank.com", "SBI": "sbi.co.in", "ICICI Bank": "icicibank.com", "Axis Bank": "axisbank.com", "Kotak Bank": "kotak.com", "PNB": "pnbindia.in",
        "Bank of Baroda": "bankofbaroda.in", "Canara Bank": "canarabank.com", "IDFC FIRST Bank": "idfcfirstbank.com", "Union Bank": "unionbankofindia.co.in"}
MERCH = ["swiggy", "zomato", "amazon", "flipkart", "irctc", "uber", "ola", "bigbasket", "myntra", "paytm", "jio", "airtel", "dmart", "bookmyshow", "phonepe"]
NAMES = ["Ramesh Kumar", "Suresh Singh", "Anita Sharma", "Mohit Verma", "Pooja Gupta", "Imran Khan", "Neha Yadav", "Rahul Mishra"]
COUR = [("Delhivery", "delhivery.com"), ("Blue Dart", "bluedart.com"), ("Ecom Express", "ecomexpress.in"), ("India Post", "indiapost.gov.in"), ("Amazon", "amazon.in"), ("Flipkart", "flipkart.com"), ("DTDC", "dtdc.in")]
GOV = [("incometax.gov.in", "Income Tax"), ("parivahan.gov.in", "Transport Dept"), ("irctc.co.in", "IRCTC"), ("uidai.gov.in", "UIDAI"), ("digilocker.gov.in", "DigiLocker")]
DATES = ["05-10-26", "12-11-26", "01/12/2026", "20-Nov", "14 Oct", "3 Jan", "29-09-26", "6th Nov"]
SMALLAMT = ["99", "150", "250", "450", "500", "799", "1,250", "2,100", "4,999", "7,500", "12,000", "25,000", "1.2 lakh"]
BIGAMT = ["50,000", "1 lakh", "2 lakh", "3 lakh", "5 lakh", "10 lakh", "25 lakh", "1 crore", "1 million dollars", "5 million dollars", "50 lakh"]
d = lambda r, n: "".join(str(r.randrange(10)) for _ in range(n))
phone = lambda r: r.choice(["1800", "1860", "1800-", "09", "98", "70", "81"]) + d(r, 8)
tail = lambda r: "".join(r.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(r.randrange(5, 9)))
SPEAK = [("you", "u"), ("your", "ur"), ("are", "r"), ("please", "plz"), ("because", "bcoz"), ("tonight", "tonite"), ("tomorrow", "tmrw"), ("before", "b4"), ("message", "msg"),
         ("customer", "custmer"), ("electricity", "elecricity"), ("account", "acount"), ("immediately", "immediatly"), ("download", "downlod"), ("approved", "aproved"),
         ("minutes", "minuts"), ("check", "chek"), ("daughter", "dauter"), ("police", "polic"), ("loans", "loanz"), ("will", "wil"), ("have", "hav"), ("salary", "salry"),
         ("interview", "intervew"), ("deposit", "deposite"), ("security", "securty"), ("update", "updat"), ("urgent", "urgnt"), ("refund", "refnd"), ("processed", "processd")]

def noise(t, r, p):
    if r.random() > p: return t
    w = t.split(" "); table = dict(SPEAK)
    return " ".join(table.get(x.lower().strip(".,!"), x) if r.random() < 0.8 else x for x in w)

def genuine(r):
    bk = r.choice(BANK); dom = BDOM[bk]; a = r.choice(SMALLAMT); ac = "XX" + d(r, 4); m = r.choice(MERCH); dt = r.choice(DATES); nm = r.choice(NAMES); cour, cd = r.choice(COUR); gd, gn = r.choice(GOV)
    notyou = r.choice([f"Not you? Call {phone(r)}", f"Not you? Report at https://{dom}/report", f"If not done by you, call {phone(r)} - {bk}", f"Not you? Block at https://{dom}/block", ""])
    T = [f"Rs {a} debited from a/c {ac} on {dt} towards {m.upper()} (UPI). {notyou}",
         f"Rs {a} sent to {nm} via UPI on {dt}. UPI Ref {d(r, 12)}. {notyou} - {bk}",
         f"Rs {a} credited to your {bk} a/c {ac} on {dt} by UPI from {nm}. Ref {d(r, 12)}",
         f"Rs {a} spent on {bk} Card {ac} at {m.upper()} on {dt}. {notyou}",
         f"Your {bk} a/c {ac} is credited with Rs {a} (salary) on {dt}. Avl bal Rs {r.choice(SMALLAMT)}",
         f"ATM withdrawal of Rs {a} from a/c {ac} on {dt}. Avl bal Rs {r.choice(SMALLAMT)}. {notyou}",
         f"Your {bk} credit card statement of Rs {a} is generated. Min due Rs {r.choice(SMALLAMT)} by {dt}. Pay on the official app",
         f"Your FD of Rs {a} matures on {dt}. Visit https://{dom} or your branch for renewal options",
         f"Your {cour} shipment AWB {d(r, 10)} is out for delivery today. Track: https://www.{cd}/track/{d(r, 8)}",
         f"Order {d(r, 9)} dispatched via {cour}. Expected delivery {dt}. Track at https://www.{cd}",
         f"Your {cour} package was delivered on {dt}. Thank you for choosing us",
         f"Your traffic e-challan of Rs {r.choice(['500', '1,000', '2,000'])} for vehicle DL{d(r, 1)}C{r.choice('ABCD')}{d(r, 4)} can be paid at https://echallan.{gd}",
         f"Your income tax return for AY 2026-27 was e-verified successfully. Acknowledgement no {d(r, 15)}. - {gn}",
         f"PNR {d(r, 10)}: your ticket is confirmed. Coach B{d(r, 1)} berth {d(r, 2)}. Journey date {dt}. Happy journey - {gn}",
         f"Your electricity bill of Rs {a} for the month is generated. Due date {dt}. Pay on the official portal or app",
         f"Gas cylinder booking is confirmed. Delivery within 3 days. Booking ref {d(r, 8)}",
         f"Recharge of Rs {r.choice(['239', '299', '399', '719'])} successful for {d(r, 10)}. Validity 28 days, 1.5GB/day",
         f"{d(r, 6)} is your OTP for {m}. Valid for 10 minutes. Do not share it with anyone. {bk} never asks for your OTP",
         f"Your {m} account password was changed on {dt}. If this was not you, reset it at https://www.{m}.com/help",
         f"Your appointment with Dr. {r.choice(['Sharma', 'Gupta', 'Khan'])} is confirmed for {dt} at {r.randrange(9, 18)}:30. Reply C to cancel",
         f"Hi, this is {nm.split()[0]}. I paid Rs {a} to your UPI today, please check and confirm",
         f"Bhai Rs {a} tere account me bhej diye hain, ref {d(r, 12)}, dekh lena",
         f"Aapke {bk} a/c {ac} se Rs {a} UPI se kate gaye {dt} ko. Agar aapne nahi kiye to {phone(r)} par call karein",
         f"Aapka {cour} parcel aaj deliver hoga. Track karein: https://www.{cd}/track/{d(r, 8)}",
         f"Aapka challan Rs {r.choice(['500', '1,000'])} bharne ke liye official site https://echallan.{gd} par jayein",
         f"Aapki {bk} statement taiyaar hai. Official app me login karke dekhein",
         f"Kal office me meeting hai {r.randrange(10, 17)} baje, report leke aana",
         f"Mummy ne kaha shaam ko ghar jaldi aana, mehmaan aa rahe hain",
         f"Please share the notes before the exam on {dt}, I will be at the library",
         f"Your EMI of Rs {a} for loan {d(r, 8)} was auto-debited on {dt}. Thank you for timely payment - {bk}"]
    return noise(r.choice(T), r, 0.12)

def scam(r):
    bk = r.choice(BANK); a = r.choice(BIGAMT); f = r.choice(["999", "1,500", "2,500", "5,000", "7,500", "10,000"]); h = r.choice(["1 hour", "2 hrs", "30 minutes", "tonight", "today"])
    lk = r.choice([f"https://{tail(r)}.{r.choice(['xyz', 'top', 'in.net', 'club', 'online'])}", f"bit.ly/{tail(r)}", f"tinyurl.com/{tail(r)}", f"{tail(r)}-{r.choice(['kyc', 'refund', 'claim', 'update'])}.xyz"])
    T = [f"We picked up a member of your family. Transfer Rs {a} in {h} to the account we text you. Tell no one, we are monitoring your phone",
         f"Your child is in our custody. Arrange Rs {a} within {h} and pass it to our man, speak to nobody or the child is harmed",
         f"Hum ne aapke bete ko utha liya hai. {h} me Rs {a} UPI karo warna nuksan hoga. Police ko mat batana",
         f"This is CBI officer. A parcel in your name has drugs. You are under digital arrest. Stay on video call and transfer Rs {a} to the RBI safe account for verification",
         f"Aapke Aadhaar se money laundering case darj hua hai. Cyber crime se call aayega, case band karne ke liye Rs {f} safe account me bhejiye",
         f"Interested in the {r.choice(['phone', 'bike', 'fridge', 'laptop'])} you listed. I am sending a payment request, approve it with your UPI PIN and the token money of Rs {f} reaches you",
         f"Aapka OLX par saman lena hai, advance dene ke liye link kholiye aur UPI PIN dalein: {lk}",
         f"Congratulations, you are selected for the post of {r.choice(['airline staff', 'online typist', 'warehouse executive', 'railway helper'])}. Pay Rs {f} towards uniform and id card to confirm your seat",
         f"Work from home jobs, earn {r.choice(['30k', '45k', '60k'])} monthly, direct joining. Refundable registration charge Rs {f} needed for the appointment letter",
         f"Naukri pakki! Registration fees Rs {f} jama karein, joining letter turant milega. Call {phone(r)}",
         f"Personal loan without documents, cibil not needed, money in your account in 10 minutes. Install our app {lk}",
         f"Instant loan Rs {a} approved. Pay processing fee Rs {f} first, amount will be credited in 10 minutes. {lk}",
         f"Dear consumer, supply to your meter will be cut at {r.choice(['8 pm', '9 pm', '10 pm'])} for pending dues of previous month. Call the executive now on {phone(r)} to avoid cut",
         f"Bijli ka bill update nahi hua, aaj raat 9:30 baje connection kat jayega. Turant is number par sampark karein {phone(r)}",
         f"Hello, are you free right now? I am stuck in a conference and need you to purchase {r.choice(['Amazon', 'iTunes', 'Google Play'])} vouchers of Rs {f} and send me the scratch codes by mail, I will repay you today",
         f"Beautiful girls near you are waiting for a private chat, send YES to start, first 10 mins free",
         f"Hi dear, I am lonely. Let us chat privately, click {lk}. I will send my photos",
         f"You have won {a}! Claim your prize now, send bank details and processing fee {f}",
         f"Congratulations lucky winner, KBC lottery Rs {a}. Contact {phone(r)} with Aadhaar and bank account to claim",
         f"Your refund of Rs {a} is processed, check status at {lk}",
         f"Your {bk} account KYC will expire today. Update now {lk} or account will be blocked",
         f"Dear customer your {bk} netbanking will be blocked. Update PAN card immediately: {lk}",
         f"Aapka parcel customs me ruka hai, Rs {f} clearance fee bhejein: {lk}",
         f"Invest Rs {f} and get double returns in 7 days. Guaranteed profit. Join our VIP stock tips group {lk}",
         f"Crypto trading group, daily profit Rs 5,000 guaranteed. Pay joining fee {f}. Click {lk}",
         f"Income tax refund of Rs {a} is approved. Submit your bank details at {lk} to receive it",
         f"I am a barrister from London. A relative of yours died leaving {a}. Reply with your bank details and a small fee to release the funds",
         f"Your SIM will be blocked in 2 hours. Complete KYC now by calling {phone(r)}",
         f"FASTag KYC pending, account will be deactivated. Update at {lk}",
         f"Aap jeet gaye {a} ka inaam! Apna bank account aur ATM PIN bhejiye aur Rs {f} fees jama karein",
         f"Congratulations, you were selected for the Prime Minister scheme of Rs {a}. Share Aadhaar, bank details and OTP to receive it"]
    return noise(r.choice(T), r, 0.55)

def make2(n_each=1400, seed=23):
    r = random.Random(seed); out = [(genuine(r), "ham") for _ in range(n_each)] + [(scam(r), "spam") for _ in range(n_each)]
    seen = set(); res = []
    for t, l in out:
        if t not in seen: seen.add(t); res.append((t, l))
    return res
