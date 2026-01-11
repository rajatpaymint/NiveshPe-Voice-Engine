import openai
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import json
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_MODIFICATIONS = 2

# -------------------------------
# Context Loader
# -------------------------------

_CONTEXT = None
def load_context():
    global _CONTEXT
    if _CONTEXT is None:
        with open("investment_context_v4.txt", "r", encoding="utf-8", errors="ignore") as f:
            _CONTEXT = f.read()
    return _CONTEXT

# -------------------------------
# Audio Helpers
# -------------------------------

def record_audio(duration=15):
    fs = 44100
    print("\n🔔 Speak now (15 seconds)...")
    time.sleep(1)

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp.name, fs, recording)
    return temp.name


def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio:
        res = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio,
            prompt="Transcribe only in Hindi (Devanagari) or English."
        )
    return res.text.strip()

# -------------------------------
# Risk Scoring Logic (Python)
# -------------------------------

def calculate_risk_score(salary, occupation, dependents, age):
    salary_points = (
        1 if salary < 25000 else
        2 if salary <= 75000 else
        3 if salary <= 150000 else
        4
    )

    occupation_points = {
        "Business": 4, "Private Job": 4, "Freelancer": 4,
        "Professional": 3, "Student": 3, "Govt Service": 3, "Public Sector Service": 3,
        "Housewife": 2, "Retired": 2, "Shopkeeper": 2, "Others": 2,
        "Agriculture": 1
    }.get(occupation, 2)

    dependent_points = 4 if dependents == 0 else 3 if dependents == 1 else 2 if dependents == 2 else 1
    age_points = 4 if age <= 25 else 3 if age <= 32 else 2 if age <= 50 else 1

    raw_score = salary_points + occupation_points + dependent_points + age_points
    risk_score = int((raw_score / 16) * 100)

    if risk_score <= 39:
        profile = "Low Risk"
    elif risk_score <= 60:
        profile = "Medium Risk"
    elif risk_score <= 80:
        profile = "High Risk"
    else:
        profile = "Ultra High Risk"

    return risk_score, profile

# -------------------------------
# LLM Call (Allocation Only)
# -------------------------------

def get_allocation(user_text, user_profile, time_horizon, previous_allocation=None):
    context = load_context()

    response = openai.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": f"""
User Profile:
{json.dumps(user_profile, ensure_ascii=False)}

Investment Time Horizon:
{time_horizon}

User Input:
{user_text}


Previous Allocation:
{json.dumps(previous_allocation, ensure_ascii=False) if previous_allocation else "NONE"}

Produce FINAL allocation JSON only.
"""
            }
        ]
    )

    return json.loads(response.output_text)

def ask_time_horizon():
    print("\nSelect investment time horizon:")
    print("1. < 1 year")
    print("2. 1–2 years")
    print("3. 2–4 years")
    print("4. 4+ years")

    while True:
        ch = input("Choice (1-4): ").strip()
        if ch == "1":
            return "<1 year"
        elif ch == "2":
            return "1-2 years"
        elif ch == "3":
            return "2-4 years"
        elif ch == "4":
            return "4+ years"


# -------------------------------
# Main Flow
# -------------------------------

def voice_ai_terminal():
    print("\n🎯 NiveshPe Voice Investment Engine\n")

    # ---- Onboarding ----
    salary = int(input("Monthly Salary (INR): "))
    print("Select Occupation:")
    occupations = [
        "Business","Private Job","Freelancer","Professional","Student",
        "Govt Service","Public Sector Service","Housewife","Retired",
        "Shopkeeper","Agriculture","Others"
    ]
    for i, o in enumerate(occupations, 1):
        print(f"{i}. {o}")
    occupation = occupations[int(input("Choice: ")) - 1]

    dependents = int(input("Number of dependents: "))
    age = int(input("Age: "))

    risk_score, risk_profile = calculate_risk_score(
        salary, occupation, dependents, age
    )

    user_profile = {
        "salary": salary,
        "occupation": occupation,
        "dependents": dependents,
        "age": age,
        "risk_score": risk_score,
        "risk_profile": risk_profile
    }

    print(f"\n📊 Risk Profile: {risk_profile} (Score: {risk_score})")

    # ---- Initial Allocation ----
    previous_allocation = None
    modification_count = 0

    while True:
        audio = record_audio()
        text = speech_to_text(audio)
        time_horizon = ask_time_horizon()

        print("\n🗣️ You said:")
        print(text)

        allocation = get_allocation(
            user_text=text,
            user_profile=user_profile,
            time_horizon=time_horizon,
            previous_allocation=previous_allocation
        )

        print("\n📊 ALLOCATION OUTPUT")
        print(json.dumps(allocation, indent=2, ensure_ascii=False))

        previous_allocation = allocation

        if modification_count >= MAX_MODIFICATIONS:
            print("\n❌ Modification limit reached. Start a new session.")
            break

        choice = input("\nDo you want modification? (Y/N): ").strip().upper()
        if choice == "N":
            print("\n✅ Allocation finalized. Thank you!")
            break

        modification_count += 1

# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    voice_ai_terminal()
