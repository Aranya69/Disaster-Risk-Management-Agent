from flask import Flask, render_template, request, redirect, url_for, session, flash
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from twilio.rest import Client
import requests
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

DEMO_USERS = {
    'demo': 'password123',
    'admin': 'admin123'
}

# API configurations
GEMINI_API_KEY = ''
TWILIO_ACCOUNT_SID = 'your-twilio-account-sid'
TWILIO_AUTH_TOKEN = 'your-twilio-auth-token'
TWILIO_PHONE_NUMBER = 'your-twilio-phone-number'
WEATHER_API_KEY = ''

# Initialize LangChain LLM with Gemini (using free tier model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_weather_data(city: str) -> dict:
    """Fetch current weather data for a given city from OpenWeatherMap API."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0)
            }
            return weather_info
        else:
            return {'error': f'Unable to fetch weather data for {city}'}
    except Exception as e:
        return {'error': f'Error fetching weather: {str(e)}'}


def send_sms_alert(phone_number: str, message: str) -> dict:
    """Send SMS alert via Twilio for weather disasters."""
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return {'success': True, 'sid': msg.sid}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_disaster_risk(weather_data: dict) -> dict:
    """Analyze weather data to determine disaster risk level."""
    try:
        risk_level = "LOW"
        alerts = []

        # Temperature analysis
        if weather_data['temperature'] > 40:
            risk_level = "EXTREME"
            alerts.append("Extreme heat warning")
        elif weather_data['temperature'] < -10:
            risk_level = "EXTREME"
            alerts.append("Extreme cold warning")

        # Wind analysis
        if weather_data['wind_speed'] > 25:
            risk_level = "EXTREME"
            alerts.append("Hurricane/Severe storm warning")
        elif weather_data['wind_speed'] > 20:
            risk_level = "HIGH"
            alerts.append("High wind warning")
        elif weather_data['wind_speed'] > 15:
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            alerts.append("Moderate wind advisory")

        # Weather condition analysis
        weather_desc = weather_data['weather'].lower()
        if any(x in weather_desc for x in ['thunderstorm', 'hurricane', 'tornado']):
            risk_level = "EXTREME"
            alerts.append("Severe storm alert")
        elif any(x in weather_desc for x in ['heavy rain', 'snow', 'blizzard']):
            if risk_level in ["LOW", "MEDIUM"]:
                risk_level = "HIGH"
            alerts.append("Severe precipitation warning")

        result = {
            'risk_level': risk_level,
            'alerts': alerts,
            'recommendation': 'SEND_ALERT' if risk_level in ['HIGH', 'EXTREME'] else 'NO_ACTION'
        }

        return result
    except Exception as e:
        return {'error': f'Error analyzing risk: {str(e)}'}


def generate_fallback_analysis(city: str, weather_data: dict, risk_analysis: dict) -> str:
    """Generate analysis when LLM fails."""
    risk = risk_analysis['risk_level']
    temp = weather_data['temperature']
    wind = weather_data['wind_speed']
    weather = weather_data['weather']
    humidity = weather_data['humidity']

    if risk == "EXTREME":
        return (f"⚠️ EXTREME WEATHER ALERT for {city}! Current conditions show {weather} with temperature at {temp}°C, "
                f"wind speeds of {wind} m/s, and humidity at {humidity}%. Immediate precautions are strongly advised. "
                f"Stay indoors, monitor local authorities, secure loose objects, and prepare emergency supplies including "
                f"water, food, flashlights, and first aid kit.")

    elif risk == "HIGH":
        return (f"⚠️ High weather risk detected in {city}. Current conditions: {weather}, {temp}°C, wind speed {wind} m/s, "
                f"humidity {humidity}%. Exercise caution and stay updated with weather forecasts. Avoid unnecessary travel "
                f"if possible and secure outdoor items. Have emergency supplies ready.")

    elif risk == "MEDIUM":
        return (f"Weather advisory for {city}: Conditions are currently {weather} with {temp}°C temperature, {wind} m/s winds, "
                f"and {humidity}% humidity. Monitor weather updates and take standard precautions. Be prepared for possible "
                f"changes in conditions.")

    else:
        return (f"Weather conditions in {city} are currently stable. Temperature: {temp}°C, Conditions: {weather}, "
                f"Wind: {wind} m/s, Humidity: {humidity}%. No immediate concerns, but stay weather-aware and check "
                f"forecasts regularly for any changes.")


def create_alert_message(city: str, weather_data: dict, risk_analysis: dict) -> str:
    """Create professional alert message for SMS."""
    alerts_text = ', '.join(risk_analysis['alerts'])
    message = f"""
🚨 WEATHER ALERT for {city.upper()} 🚨

Risk Level: {risk_analysis['risk_level']}
Alerts: {alerts_text}

Current Conditions:
• Temperature: {weather_data['temperature']}°C
• Wind Speed: {weather_data['wind_speed']} m/s
• Weather: {weather_data['weather']}

Safety Recommendations:
• Stay indoors if possible
• Monitor local weather updates
• Prepare emergency supplies
• Follow official evacuation orders if issued

Stay safe!
    """.strip()
    return message


def monitor_weather_with_llm(city: str, phone_number: str) -> dict:
    """Monitor weather and handle alerts with LangChain integration."""

    # Step 1: Get weather data
    weather_data = get_weather_data(city)
    if 'error' in weather_data:
        return {'error': weather_data['error']}

    # Step 2: Analyze disaster risk
    risk_analysis = analyze_disaster_risk(weather_data)
    if 'error' in risk_analysis:
        return {'error': risk_analysis['error']}

    # Step 3: Use LLM to enhance analysis with error handling
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional weather disaster expert. Provide clear, concise risk assessment and actionable safety recommendations in 3-4 sentences."),
            ("human", """Analyze this weather situation and provide professional guidance:

City: {city}
Temperature: {temp}°C (Feels like: {feels_like}°C)
Wind Speed: {wind} m/s
Humidity: {humidity}%
Weather: {weather}
Risk Level: {risk_level}
Active Alerts: {alerts}

Provide a brief professional weather assessment with specific safety recommendations.""")
        ])

        chain = prompt | llm | StrOutputParser()

        llm_analysis = chain.invoke({
            "city": city,
            "temp": weather_data['temperature'],
            "feels_like": weather_data['feels_like'],
            "wind": weather_data['wind_speed'],
            "humidity": weather_data['humidity'],
            "weather": weather_data['weather'],
            "risk_level": risk_analysis['risk_level'],
            "alerts": ', '.join(risk_analysis['alerts']) if risk_analysis['alerts'] else 'None'
        })

        # Ensure non-empty response
        if not llm_analysis or llm_analysis.strip() == "":
            print("LLM returned empty response, using fallback")
            llm_analysis = generate_fallback_analysis(city, weather_data, risk_analysis)
        else:
            print(f"LLM Analysis: {llm_analysis[:100]}...")  # Debug log

    except Exception as e:
        print(f"LLM Error: {str(e)}")
        llm_analysis = generate_fallback_analysis(city, weather_data, risk_analysis)

    result = {
        'city': city,
        'weather_data': weather_data,
        'risk_analysis': risk_analysis,
        'llm_analysis': llm_analysis,
        'sms_sent': False
    }

    # Step 4: Send SMS if high/extreme risk
    if risk_analysis['risk_level'] in ['HIGH', 'EXTREME']:
        alert_message = create_alert_message(city, weather_data, risk_analysis)
        sms_result = send_sms_alert(phone_number, alert_message)
        result['sms_sent'] = sms_result.get('success', False)
        result['sms_result'] = sms_result

    return result


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in DEMO_USERS and DEMO_USERS[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])


@app.route('/monitor', methods=['POST'])
@login_required
def monitor_weather():
    city = request.form.get('city')
    phone_number = request.form.get('phone_number')

    if not city or not phone_number:
        flash('Please provide both city and phone number!', 'error')
        return redirect(url_for('dashboard'))

    try:
        result = monitor_weather_with_llm(city, phone_number)

        if 'error' in result:
            flash(f'Error: {result["error"]}', 'error')
            return redirect(url_for('dashboard'))

        return render_template('results.html',
                             city=city,
                             phone_number=phone_number,
                             result=result)

    except Exception as e:
        print(f"Error during monitoring: {str(e)}")  # Debug log
        flash(f'Error during monitoring: {str(e)}', 'error')
        return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
