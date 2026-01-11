from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import json
import os
import tempfile
import base64
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('niveshpe_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
logger.info("="*80)
logger.info("NiveshPe Voice Investment Engine Started")
logger.info("="*80)

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
# Risk Scoring Logic
# -------------------------------

def calculate_risk_score(salary, occupation, dependents, age):
    logger.info("-" * 80)
    logger.info("RISK CALCULATION STARTED")
    logger.info(f"Input: Salary={salary}, Occupation={occupation}, Dependents={dependents}, Age={age}")

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

    logger.info(f"Points Breakdown: Salary={salary_points}, Occupation={occupation_points}, Dependents={dependent_points}, Age={age_points}")
    logger.info(f"Raw Score: {raw_score}/16 → Risk Score: {risk_score}/100")
    logger.info(f"Risk Profile: {profile}")
    logger.info("RISK CALCULATION COMPLETED")
    logger.info("-" * 80)

    return risk_score, profile

# -------------------------------
# Allocation Validation Function
# -------------------------------

def validate_allocation(allocation, time_horizon):
    """
    Validates allocation against strict bounds.
    Returns: (is_valid, violations, calculated_percentages)
    """
    # Define fund composition mappings
    fund_compositions = {
        "MULTICAP": {"equity": 100, "debt": 0},
        "FLEXICAP": {"equity": 100, "debt": 0},
        "LARGE_CAP": {"equity": 100, "debt": 0},
        "LARGE_MID_CAP": {"equity": 100, "debt": 0},
        "LARGE&MID_CAP": {"equity": 100, "debt": 0},
        "AGGRESSIVE_HYBRID": {"equity": 75, "debt": 25},
        "BALANCED_ADVANTAGE": {"equity": 50, "debt": 50},
        "CONSERVATIVE_HYBRID": {"equity": 20, "debt": 80},
        "DEBT": {"equity": 0, "debt": 100},
        "LIQUID": {"equity": 0, "debt": 100},
        "ARBITRAGE": {"equity": 0, "debt": 100},
        "GOLD": {"equity": 0, "debt": 0},
        "SILVER": {"equity": 0, "debt": 0}
    }

    # Calculate actual allocation percentages
    total_equity = 0
    total_debt = 0
    total_precious_metals = 0

    for fund in allocation.get('allocation', []):
        category = fund.get('fund_category', '').upper()
        percentage = fund.get('percentage', 0)

        if category in ["GOLD", "SILVER"]:
            total_precious_metals += percentage
        else:
            comp = fund_compositions.get(category, {"equity": 0, "debt": 0})
            total_equity += (percentage * comp['equity']) / 100
            total_debt += (percentage * comp['debt']) / 100

    # Define expected bounds based on risk profile
    risk_profile = allocation.get('risk_profile', '')
    bounds = {
        "Low Risk": {"equity_min": 34, "equity_max": 62, "debt_min": 28, "debt_max": 56},
        "Medium Risk": {"equity_min": 62, "equity_max": 72, "debt_min": 18, "debt_max": 28},
        "High Risk": {"equity_min": 72, "equity_max": 83, "debt_min": 8, "debt_max": 18},
        "Ultra High Risk": {"equity_min": 83, "equity_max": 90, "debt_min": 0, "debt_max": 8}
    }

    expected = bounds.get(risk_profile, {"equity_min": 0, "equity_max": 100, "debt_min": 0, "debt_max": 100})

    # Validation checks
    violations = []
    total = total_equity + total_debt + total_precious_metals

    # Check total = 100%
    if abs(total - 100) > 0.1:
        violations.append(f"Total allocation is {total:.2f}% (should be 100%)")

    # Check equity bounds
    if total_equity < expected['equity_min']:
        violations.append(f"Equity {total_equity:.2f}% is BELOW minimum {expected['equity_min']}%")
    elif total_equity > expected['equity_max']:
        violations.append(f"Equity {total_equity:.2f}% EXCEEDS maximum {expected['equity_max']}%")

    # Check debt bounds
    if total_debt < expected['debt_min']:
        violations.append(f"Debt {total_debt:.2f}% is BELOW minimum {expected['debt_min']}%")
    elif total_debt > expected['debt_max']:
        violations.append(f"Debt {total_debt:.2f}% EXCEEDS maximum {expected['debt_max']}%")

    # Check precious metals for time horizon > 2 years
    if time_horizon in ['2-4 years', '4+ years']:
        if total_precious_metals != 10:
            violations.append(f"Precious metals {total_precious_metals:.2f}% should be exactly 10%")

    is_valid = len(violations) == 0

    calculated_percentages = {
        'equity': total_equity,
        'debt': total_debt,
        'precious_metals': total_precious_metals,
        'total': total,
        'expected_bounds': expected
    }

    return is_valid, violations, calculated_percentages

# -------------------------------
# LLM Call (Allocation) with Retry
# -------------------------------

def get_allocation(user_text, user_profile, time_horizon, previous_allocation=None):
    logger.info("=" * 80)
    logger.info("ALLOCATION GENERATION STARTED (WITH AUTO-RETRY VALIDATION)")
    logger.info("=" * 80)

    context = load_context()
    logger.info(f"Context loaded: {len(context)} characters")
    logger.info(f"User Profile: {json.dumps(user_profile, ensure_ascii=False)}")
    logger.info(f"Time Horizon: {time_horizon}")
    logger.info(f"User Input (Transcribed): {user_text}")
    logger.info(f"Previous Allocation: {'Yes' if previous_allocation else 'No (First allocation)'}")

    MAX_ATTEMPTS = 3
    last_allocation = None
    last_violations = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info("=" * 80)
        logger.info(f"ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        logger.info("=" * 80)

        # Build user message (with correction prompt if retry)
        if attempt == 1:
            user_message = f"""
User Profile:
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

Investment Time Horizon:
{time_horizon}

User Input:
{user_text}

Previous Allocation:
{json.dumps(previous_allocation, ensure_ascii=False, indent=2) if previous_allocation else "NONE"}

Produce FINAL allocation JSON only.
"""
        else:
            # Correction prompt for retries
            violations_text = "\n".join([f"  - {v}" for v in last_violations])
            user_message = f"""
CORRECTION REQUIRED - Your previous allocation had the following violations:

{violations_text}

You MUST fix these violations by adjusting fund percentages.

Remember the fund compositions:
- Pure Equity funds (Multicap, Flexicap, Large Cap, Large&Mid Cap): 100% equity, 0% debt
- Aggressive Hybrid: 75% equity, 25% debt
- Balanced Advantage: 50% equity, 50% debt
- Conservative Hybrid: 20% equity, 80% debt
- Debt/Liquid/Arbitrage: 0% equity, 100% debt
- Gold/Silver: Precious metals (not equity/debt)

User Profile:
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

Investment Time Horizon:
{time_horizon}

User Input:
{user_text}

Previous Allocation:
{json.dumps(previous_allocation, ensure_ascii=False, indent=2) if previous_allocation else "NONE"}

CREATE A NEW ALLOCATION that fixes the violations above. Produce FINAL allocation JSON only.
"""

        logger.info("-" * 80)
        logger.info(f"REQUEST TO GPT-4 (Attempt {attempt})")
        logger.info("-" * 80)
        if attempt > 1:
            logger.info("CORRECTION PROMPT: Fixing violations from previous attempt")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )

            allocation_json = response.choices[0].message.content
            logger.info("-" * 80)
            logger.info("RESPONSE FROM GPT-4")
            logger.info("-" * 80)
            logger.info(f"Raw Response:\n{allocation_json}")
            logger.info("-" * 80)

            allocation = json.loads(allocation_json)
            last_allocation = allocation

            # Log key details of the allocation
            logger.info("ALLOCATION ANALYSIS")
            logger.info("-" * 80)
            logger.info(f"Risk Profile: {allocation.get('risk_profile', 'N/A')}")
            logger.info(f"Risk Score: {allocation.get('risk_score', 'N/A')}")
            logger.info(f"Goal: {allocation.get('goal', 'N/A')}")
            logger.info(f"Time Horizon (months): {allocation.get('time_horizon_months', 'N/A')}")
            logger.info(f"Blended 3Y CAGR: {allocation.get('blended_3y_cagr', 'N/A')}%")
            logger.info(f"Number of Funds: {len(allocation.get('allocation', []))}")

            # Log each fund
            logger.info("-" * 80)
            logger.info("FUND BREAKDOWN")
            logger.info("-" * 80)
            for idx, fund in enumerate(allocation.get('allocation', []), 1):
                logger.info(f"Fund {idx}: {fund.get('fund_name', 'N/A')} ({fund.get('fund_category', 'N/A')})")
                logger.info(f"  → Allocation: {fund.get('percentage', 0)}%")
                logger.info(f"  → 3Y CAGR: {fund.get('cagr_3y', 'N/A')}%")
                logger.info(f"  → ISIN: {fund.get('isin', 'N/A')}")

            # Validate allocation using the validation function
            logger.info("-" * 80)
            logger.info("ASSET ALLOCATION VALIDATION")
            logger.info("-" * 80)

            is_valid, violations, calc_percentages = validate_allocation(allocation, time_horizon)

            logger.info(f"Calculated Asset Allocation:")
            logger.info(f"  Equity: {calc_percentages['equity']:.2f}%")
            logger.info(f"  Debt: {calc_percentages['debt']:.2f}%")
            logger.info(f"  Precious Metals: {calc_percentages['precious_metals']:.2f}%")
            logger.info(f"  Total: {calc_percentages['total']:.2f}%")

            expected = calc_percentages['expected_bounds']
            logger.info(f"Expected Bounds for {allocation.get('risk_profile', 'N/A')}:")
            logger.info(f"  Equity: {expected['equity_min']}-{expected['equity_max']}%")
            logger.info(f"  Debt: {expected['debt_min']}-{expected['debt_max']}%")
            logger.info(f"  Precious Metals: 10% (for horizon > 2 years)")

            if is_valid:
                logger.info("=" * 80)
                logger.info(f"✅ VALIDATION PASSED ON ATTEMPT {attempt}/{MAX_ATTEMPTS}")
                logger.info("=" * 80)
                logger.info("✓ All allocation bounds validated successfully")
                logger.info("-" * 80)

                # Check for precious metals
                fund_categories = [f.get('fund_category', '').upper() for f in allocation.get('allocation', [])]
                has_gold = any('GOLD' in cat for cat in fund_categories)
                has_silver = any('SILVER' in cat for cat in fund_categories)
                logger.info(f"Precious Metals Check: Gold={has_gold}, Silver={has_silver}")
                logger.info("-" * 80)

                logger.info("ALLOCATION GENERATION COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)
                return allocation
            else:
                # Validation failed - log violations and prepare for retry
                logger.warning("=" * 80)
                logger.warning(f"❌ VALIDATION FAILED ON ATTEMPT {attempt}/{MAX_ATTEMPTS}")
                logger.warning("=" * 80)
                logger.warning("⚠️  ALLOCATION BOUND VIOLATIONS DETECTED:")
                for violation in violations:
                    logger.warning(f"  ✗ {violation}")
                logger.warning("-" * 80)

                last_violations = violations

                if attempt < MAX_ATTEMPTS:
                    logger.warning(f"🔄 Retrying... (Attempt {attempt + 1}/{MAX_ATTEMPTS})")
                else:
                    logger.error("=" * 80)
                    logger.error("❌ MAX RETRY ATTEMPTS REACHED")
                    logger.error("=" * 80)
                    logger.error(f"All {MAX_ATTEMPTS} attempts failed validation.")
                    logger.error("Returning last allocation despite violations.")
                    logger.error("=" * 80)

                    # Check for precious metals even on failure
                    fund_categories = [f.get('fund_category', '').upper() for f in allocation.get('allocation', [])]
                    has_gold = any('GOLD' in cat for cat in fund_categories)
                    has_silver = any('SILVER' in cat for cat in fund_categories)
                    logger.info(f"Precious Metals Check: Gold={has_gold}, Silver={has_silver}")

                    return last_allocation

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"ERROR IN ATTEMPT {attempt}")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")

            if attempt < MAX_ATTEMPTS:
                logger.error(f"Retrying... (Attempt {attempt + 1}/{MAX_ATTEMPTS})")
                continue
            else:
                logger.error("=" * 80)
                logger.error("ERROR IN ALLOCATION GENERATION")
                logger.error("=" * 80)
                raise

    # Should never reach here, but just in case
    logger.error("Unexpected: Loop completed without returning allocation")
    if last_allocation:
        return last_allocation
    raise Exception("Failed to generate allocation after all attempts")

# -------------------------------
# API Routes
# -------------------------------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/calculate-risk', methods=['POST'])
def api_calculate_risk():
    try:
        data = request.json
        salary = int(data['salary'])
        occupation = data['occupation']
        dependents = int(data['dependents'])
        age = int(data['age'])

        risk_score, risk_profile = calculate_risk_score(salary, occupation, dependents, age)

        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_profile': risk_profile,
            'user_profile': {
                'salary': salary,
                'occupation': occupation,
                'dependents': dependents,
                'age': age,
                'risk_score': risk_score,
                'risk_profile': risk_profile
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    try:
        logger.info("=" * 80)
        logger.info("TRANSCRIPTION REQUEST RECEIVED")
        logger.info("=" * 80)

        data = request.json
        audio_base64 = data['audio']

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64.split(',')[1] if ',' in audio_base64 else audio_base64)
        logger.info(f"Audio size: {len(audio_bytes)} bytes")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        logger.info(f"Audio saved to temp file: {temp_path}")
        logger.info("Sending to OpenAI Whisper for transcription...")

        # Transcribe
        with open(temp_path, "rb") as audio:
            res = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                prompt="Transcribe only in Hindi (Devanagari) or English."
            )

        transcribed_text = res.text.strip()
        logger.info("-" * 80)
        logger.info("TRANSCRIPTION RESULT")
        logger.info("-" * 80)
        logger.info(f"Transcribed Text: {transcribed_text}")
        logger.info(f"Text Length: {len(transcribed_text)} characters")
        logger.info("=" * 80)

        # Clean up
        os.unlink(temp_path)

        return jsonify({
            'success': True,
            'text': transcribed_text
        })
    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRANSCRIPTION ERROR")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80)
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/allocate', methods=['POST'])
def api_allocate():
    try:
        logger.info("\n" + "=" * 80)
        logger.info("API ALLOCATION REQUEST RECEIVED")
        logger.info("=" * 80)

        data = request.json
        user_text = data['user_text']
        user_profile = data['user_profile']
        time_horizon = data['time_horizon']
        previous_allocation = data.get('previous_allocation', None)

        allocation = get_allocation(user_text, user_profile, time_horizon, previous_allocation)

        # Detect language from user input
        detected_language = detect_language(user_text)

        logger.info(f"Detected Language: {detected_language}")
        logger.info("API ALLOCATION REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80 + "\n")

        return jsonify({
            'success': True,
            'allocation': allocation,
            'detected_language': detected_language
        })
    except Exception as e:
        logger.error("API ALLOCATION REQUEST FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/text-to-speech', methods=['POST'])
def api_text_to_speech():
    try:
        logger.info("=" * 80)
        logger.info("TEXT-TO-SPEECH REQUEST RECEIVED")
        logger.info("=" * 80)

        data = request.json
        text = data['text']
        language = data.get('language', 'english')

        logger.info(f"Text Length: {len(text)} characters")
        logger.info(f"Language: {language}")

        # Choose voice based on language
        # For OpenAI TTS, we'll use 'nova' for female voice
        # Note: OpenAI TTS doesn't have native Hindi support, but can pronounce Hindi text
        voice = "nova"  # Female voice

        # Generate speech
        response = openai.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=text,
            speed=0.95  # Slightly slower for clarity
        )

        # Convert to base64 for transmission
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        logger.info(f"Audio generated: {len(audio_bytes)} bytes")
        logger.info("TEXT-TO-SPEECH COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return jsonify({
            'success': True,
            'audio': f"data:audio/mp3;base64,{audio_base64}",
            'language': language
        })
    except Exception as e:
        logger.error("TEXT-TO-SPEECH FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80)
        return jsonify({'success': False, 'error': str(e)}), 400

def detect_language(text):
    """
    Detect if text is Hindi (Devanagari) or English
    """
    # Check for Devanagari Unicode range (Hindi)
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])

    if total_chars == 0:
        return 'english'

    hindi_ratio = hindi_chars / total_chars

    # If more than 30% Hindi characters, consider it Hindi
    if hindi_ratio > 0.3:
        return 'hindi'
    else:
        return 'english'

# -------------------------------
# Conversation Management System
# -------------------------------

@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """
    Starts a new conversation after risk profiling.
    Returns greeting message with TTS.
    """
    try:
        data = request.json
        user_profile = data.get('user_profile')

        if not user_profile:
            return jsonify({'success': False, 'error': 'User profile required'}), 400

        risk_profile = user_profile.get('risk_profile', 'Medium Risk')

        logger.info("="*80)
        logger.info("CONVERSATION STARTED")
        logger.info(f"Risk Profile: {risk_profile}")

        # Load context
        context = load_context()

        # Generate greeting using GPT-4
        prompt = f"""
{context}

User has completed risk profiling:
Risk Profile: {risk_profile}
Risk Score: {user_profile.get('risk_score', 50)}

Generate a warm, conversational greeting as per STAGE 1: PERSONALIZED GREETING rules.

Return ONLY valid JSON:
{{
  "stage": "greeting",
  "bot_message": "...",
  "bot_message_hindi": "..." (if user might speak Hindi, provide Hindi version too),
  "needs_response": true
}}
"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        bot_message = result.get('bot_message', '')

        logger.info(f"Greeting Generated: {bot_message}")

        # Generate TTS audio
        detected_language = 'english'  # Default for greeting
        voice = "nova"  # Female voice

        tts_response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=bot_message,
            speed=0.95
        )

        audio_bytes = tts_response.content
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        result['audio'] = f"data:audio/mp3;base64,{audio_base64}"
        result['success'] = True

        logger.info("Greeting audio generated successfully")
        logger.info("="*80)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conversation/continue', methods=['POST'])
def continue_conversation():
    """
    Handles multi-turn conversation.
    Extracts info, asks questions, or generates allocation.
    """
    try:
        data = request.json
        user_text = data.get('user_text', '')
        user_profile = data.get('user_profile')
        conversation_history = data.get('conversation_history', [])
        previous_allocation = data.get('previous_allocation')

        if not user_text or not user_profile:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        logger.info("="*80)
        logger.info("CONVERSATION TURN")
        logger.info(f"User Input: {user_text}")
        logger.info(f"Conversation Turn: {len(conversation_history) + 1}")

        # Load context
        context = load_context()

        # Build conversation context
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        ])

        # Determine if this is a modification request
        is_modification = previous_allocation is not None

        if is_modification:
            logger.info("Handling modification request")
            prompt = f"""
{context}

User Profile:
{json.dumps(user_profile, indent=2)}

Previous Allocation:
{json.dumps(previous_allocation, indent=2)}

User wants to modify: "{user_text}"

Parse the modification request and generate a new allocation following all rules.
Return the FULL allocation JSON as per FINAL OUTPUT FORMAT.
"""
        else:
            logger.info("Processing conversation turn for new allocation")
            prompt = f"""
{context}

User Profile:
{json.dumps(user_profile, indent=2)}

Conversation History:
{conversation_context}

Latest User Input: "{user_text}"

Follow CONVERSATIONAL FLOW MANAGEMENT rules:
1. Extract all available information from user input
2. Check what's still missing (goal, time_horizon)
3. If time_horizon is missing, ask for it
4. If everything is available, generate allocation
5. Return appropriate JSON based on stage

Return ONLY valid JSON in one of these formats:

If need to ask question:
{{
  "stage": "questioning",
  "bot_message": "...",
  "question_type": "time_horizon" | "goal" | "preferences",
  "needs_response": true,
  "collected_info": {{
    "goal": "..." or null,
    "time_horizon": "..." or null,
    "preferences": {{...}}
  }}
}}

If ready to allocate:
{{
  "stage": "ready_to_allocate",
  "proceed_to_allocation": true,
  "collected_info": {{
    "goal": "...",
    "time_horizon": "...",
    "preferences": {{...}}
  }}
}}
"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        logger.info(f"Stage: {result.get('stage', 'unknown')}")

        # If ready to allocate, generate the allocation
        if result.get('proceed_to_allocate'):
            logger.info("Proceeding to allocation generation")
            collected_info = result.get('collected_info', {})
            time_horizon = collected_info.get('time_horizon', '4+ years')

            # Generate allocation
            allocation = get_allocation(
                user_text=user_text,
                user_profile=user_profile,
                time_horizon=time_horizon,
                previous_allocation=previous_allocation
            )

            if allocation:
                result = {
                    'success': True,
                    'stage': 'allocation_generated',
                    'allocation': allocation,
                    'detected_language': detect_language(user_text)
                }
            else:
                result = {
                    'success': False,
                    'error': 'Failed to generate allocation after maximum attempts'
                }

        # If bot needs to ask question, generate TTS
        elif result.get('needs_response') and result.get('bot_message'):
            bot_message = result['bot_message']
            logger.info(f"Bot Question: {bot_message}")

            # Detect language
            detected_language = detect_language(user_text)
            voice = "nova"

            # Generate TTS
            tts_response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=bot_message,
                speed=0.95
            )

            audio_bytes = tts_response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            result['audio'] = f"data:audio/mp3;base64,{audio_base64}"
            result['success'] = True

        logger.info("CONVERSATION TURN COMPLETED")
        logger.info("="*80)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
