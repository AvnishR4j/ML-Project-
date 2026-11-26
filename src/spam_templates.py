"""Generated Indian-style spam and GENUINE service messages (English + Hinglish), training only.
Genuine transactional messages (bank alerts, OTP notices, bills, delivery updates, family money talk) are included on purpose:
without them the model flags real bank alerts as spam because they contain money and account words."""
import random
AMT = ["500", "1,200", "2,300", "4,999", "9,800", "15,000", "32,500", "1.5 lakh", "2 lakh", "3 lakh", "5 lakh", "10 lakh", "25 lakh", "1 crore", "2 crore", "5 crore"]
SMALL = ["25", "49", "99", "150", "500", "999", "1500", "2000", "3000", "5000", "10000"]
MERCH = ["Amazon", "Zomato", "Swiggy", "Flipkart", "BigBasket", "IRCTC", "Uber", "Myntra", "a fuel pump", "D-Mart", "Jio recharge"]
DATE = ["5th", "10th", "12 Nov", "20 Dec", "Friday", "Monday", "tomorrow", "3 Jan"]
BANK = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Kotak", "Bank of Baroda", "Paytm Payments Bank"]
SVC = ["login", "payment", "transaction", "net banking", "UPI", "your account", "card payment"]
NM = ["Ramesh", "Suresh", "Anita", "Mohit", "Pooja", "Imran"]
d4 = lambda r: str(r.randrange(1000, 9999)); d6 = lambda r: str(r.randrange(100000, 999999))

def genuine(r):
    a, b, m, dt, bk, sv = r.choice(AMT), r.choice(AMT), r.choice(MERCH), r.choice(DATE), r.choice(BANK), r.choice(SVC)
    T = [f"Rs {a} debited from your account ending {d4(r)} at {m} on {dt}", f"Rs {a} credited to your account ending {d4(r)}. Available balance Rs {b}",
         f"{d6(r)} is your OTP for {sv}. Do not share it with anyone", f"Your OTP is {d6(r)}, valid for 10 minutes. {bk} never asks for your OTP or PIN",
         f"Never share this code with anyone, including {bk} staff. OTP: {d6(r)}", f"Your bill of Rs {a} is due on {dt}. Pay using the official {bk} app",
         f"Your EMI of Rs {a} was deducted successfully on {dt}", f"Payment of Rs {a} received. Thank you. Your plan is active until {dt}",
         f"Your order from {m} was delivered today. Thank you for shopping with us", f"Your request has been registered. Reference number {r.randrange(10000, 99999)}",
         f"Your ride is arriving in {r.randrange(2, 9)} minutes. Driver {r.choice(NM)}", f"Your {bk} statement is ready. Log in to the official app to view it",
         f"Salary of Rs {a} has been credited to your account on {dt}", f"UPI transfer of Rs {a} to {r.choice(NM)} was successful. If this was not you, contact {bk} customer care",
         f"Aapke account se Rs {a} {r.choice(['UPI', 'ATM', 'card'])} se kate gaye. Agar aapne nahi kiye to {bk} branch mein sampark karein", f"Aapka order {dt} tak pahunch jayega, tracking app mein dekh sakte hain",
         f"Aapka bill Rs {a} baki hai, last date {dt}. Official app se bharein", f"Refund Rs {a} aapke original payment method mein 5-7 din mein aa jayega",
         f"Mummy ne {a} ka paisa bank mein jama kiya hai, kal passbook update karwana hai", f"Bhai {a} rupaye UPI se bhej diye, check kar lena",
         f"Papa ko {a} ka loan lena hai ghar ke liye, kal bank jaake baat karenge", f"College fees {a} ki hai, last date {dt} hai, papa ko bata diya",
         f"Did the {a} project get approved? Let me know", f"We got a grant of {a} for the lab, principal signed the paper today",
         f"Subsidy ka form {dt} tak jama karna hai tehsil mein, saath chalte hain", f"Your KYC documents were verified successfully. No further action needed",
         f"Dear customer, your {bk} cheque book has been dispatched and will arrive in 5 days", f"Your appointment is confirmed for {dt} at {r.randrange(9, 17)}:00"]
    return r.choice(T)

def scam(r):
    a, f, bk, dt = r.choice(AMT), r.choice(SMALL), r.choice(BANK), r.choice(DATE)
    sch = r.choice(["PM Awas Yojana", "PM Kisan Yojana", "government subsidy scheme", "sarkari yojana", "Mudra loan scheme", "KBC lucky draw", "Diwali lucky draw", "scholarship scheme", "gas subsidy", "income tax refund"])
    link = r.choice(["", "", "", " bit.ly/" + "".join(r.choice("abcdefgh12345") for _ in range(6)), f" {bk.lower().replace(' ', '')}-kyc-update.in", " tinyurl.com/claim" + str(r.randrange(10, 99))])
    hook = r.choice([f"I got {a} {sch.split()[-1]}", f"You have won Rs {a} in {sch}", f"Aapko {sch} mein {a} mila hai", f"Mujhe {sch} se {a} milne wale hain", f"Congratulations! You are selected for Rs {a}",
                     f"Your {bk} account KYC has expired", f"Aapka {bk} account block ho jayega", f"Your account will be suspended today", f"Loan of Rs {a} is pre-approved for you",
                     f"Refund of Rs {a} is pending in your name", f"Aapka parcel customs mein ruka hai", f"Earn Rs {f} daily from home", f"Aapka {sch} ka paisa ruka hua hai", f"Aap lucky winner hain, {a} ka inaam"])
    ask = r.choice([f"Send your bank account number and PIN to receive it{link}", f"Share the OTP you received to continue{link}", f"Pay a processing fee of Rs {f} first{link}", f"Click{link or ' this link'} and enter your card number and CVV",
                    f"Verify your Aadhaar and PAN now{link}", f"Apni bank details aur OTP bhejiye{link}", f"Processing fee Rs {f} abhi jama karein{link}", f"Abhi{link or ' is link'} par click karke details verify karein warna band ho jayega",
                    f"Call this number immediately to claim{link}", "bank details dene hain", "ke liye pehle apna account number aur ATM PIN batayein", "Banking details dene padenge", f"Send Rs {f} registration charge to confirm your seat"])
    return r.choice([f"{hook}. {ask}", f"{hook}, {ask[0].lower() + ask[1:]}", f"{hook} so I need to give {r.choice(['bank details', 'banking details', 'my account details'])}", f"{hook}. {ask}. Urgent"])

def make(n_each=1500, seed=11):
    r = random.Random(seed); out = [(genuine(r), "ham") for _ in range(n_each)] + [(scam(r), "spam") for _ in range(n_each)]
    seen = set(); res = []
    for t, l in out:
        if t not in seen: seen.add(t); res.append((t, l))
    return res
