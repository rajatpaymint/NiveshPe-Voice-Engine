# NiveshPe Voice Investment Engine

A voice-powered AI investment recommendation system for Indian investors, featuring a modern web interface with real-time voice transcription and intelligent fund allocation.

## Features

- **User Onboarding**: Collect user profile (salary, occupation, dependents, age)
- **Risk Assessment**: Automatic risk score calculation (0-100 scale)
- **Voice Input**: Record investment goals in English or Hindi
- **AI-Powered Allocation**: SEBI-safe fund recommendations based on user profile
- **Interactive Modifications**: Up to 2 modifications allowed per session
- **Responsive Design**: Modern, mobile-friendly interface

## Project Structure

```
VoiceClaude/
├── app.py                      # Flask backend API
├── voice_v3.py                 # Original CLI version (reference)
├── investment_context_v4.txt   # Investment recommendation rules
├── requirements.txt            # Python dependencies
├── static/
│   └── index.html             # Frontend web interface
└── README.md                   # This file
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Modern web browser with microphone access
- Internet connection

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "/Users/rajatyadav/Documents/NiveshPe Docs/Engines/VoiceClaude"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to: `http://localhost:5000`

3. **Allow microphone access**
   When prompted, allow your browser to access the microphone

## Usage Flow

### Step 1: User Profile
1. Enter your monthly salary (INR)
2. Select your occupation from the dropdown
3. Enter number of dependents
4. Enter your age
5. Click "Calculate Risk Profile"

### Step 2: Voice Input
1. View your calculated risk profile
2. Select investment time horizon
3. Click the microphone button to record (15 seconds)
4. Speak your investment goals in English or Hindi
5. Wait for transcription to complete
6. Click "Generate Allocation"

### Step 3: View Recommendations
1. Review your personalized fund allocation
2. See historical 3Y CAGR for each fund
3. View blended portfolio return
4. Choose to:
   - **Finalize**: Accept the allocation
   - **Modify**: Record new requirements (max 2 modifications)

## API Endpoints

### POST `/api/calculate-risk`
Calculate user risk profile

**Request Body:**
```json
{
  "salary": 50000,
  "occupation": "Private Job",
  "dependents": 2,
  "age": 30
}
```

**Response:**
```json
{
  "success": true,
  "risk_score": 56,
  "risk_profile": "Medium Risk",
  "user_profile": { ... }
}
```

### POST `/api/transcribe`
Transcribe audio to text

**Request Body:**
```json
{
  "audio": "data:audio/wav;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "text": "I want to invest for my child's education"
}
```

### POST `/api/allocate`
Generate fund allocation

**Request Body:**
```json
{
  "user_text": "I want to invest for my child's education",
  "user_profile": { ... },
  "time_horizon": "4+ years",
  "previous_allocation": null
}
```

**Response:**
```json
{
  "success": true,
  "allocation": {
    "risk_profile": "Medium Risk",
    "risk_score": 56,
    "goal": "Child's education",
    "time_horizon_months": 48,
    "blended_3y_cagr": 22.5,
    "description": "...",
    "allocation": [
      {
        "fund_category": "MULTICAP",
        "fund_name": "Kotak Multicap",
        "isin": "INF174KA1HS9",
        "percentage": 40,
        "cagr_3y": 25.79
      }
    ]
  }
}
```

## Investment Rules

The system follows strict SEBI-safe guidelines:

- **Risk Profiles**: Low Risk, Medium Risk, High Risk, Ultra High Risk
- **Asset Classes**: Equity, Debt, Precious Metals
- **Time Horizons**: <1 year, 1-2 years, 2-4 years, 4+ years
- **Fund Universe**: 11 pre-selected funds with verified historical returns
- **Modifications**: Maximum 2 modifications per session
- **Fund Limit**: 4 funds initially, 5 funds after modifications

## Compliance

- No guaranteed returns
- No future return claims
- Historical returns clearly marked
- SEBI-safe language and disclaimers
- No internet data fetching for fund information

## Troubleshooting

### Microphone not working
- Ensure browser has microphone permissions
- Check system microphone settings
- Try using HTTPS (required for some browsers)

### OpenAI API errors
- Verify API key is set correctly
- Check API key has sufficient credits
- Ensure you're using supported models (gpt-4o-transcribe, gpt-5-nano)

### Flask server not starting
- Check if port 5000 is available
- Verify all dependencies are installed
- Check Python version (3.8+)

## Development

### Running in Debug Mode
The Flask app runs in debug mode by default. To disable:
```python
# In app.py, change:
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Testing API Endpoints
Use curl or Postman to test individual endpoints:
```bash
curl -X POST http://localhost:5000/api/calculate-risk \
  -H "Content-Type: application/json" \
  -d '{"salary":50000,"occupation":"Private Job","dependents":2,"age":30}'
```

## Original CLI Version

The original command-line version (`voice_v3.py`) is included for reference. To run it:
```bash
python voice_v3.py
```

## License

Proprietary - NiveshPe

## Support

For issues or questions, contact the development team.
