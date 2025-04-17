from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
import re
from recommendation import get_recommendation
import google.generativeai as genai
import os
import numpy as np
import json
from datetime import datetime, timedelta
import difflib
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
assistant = pc.assistant.Assistant(assistant_name="rag1")

# Initialize Gemini AI
genai.configure(api_key=os.getenv('gemini_api_key'))

# Get Gemini model for translation
translation_model = genai.GenerativeModel('Gemini-2.0-Flash')
# Get Gemini model for calculations and analysis
calculation_model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Global session tracking for user conversations
user_sessions = {}

# Global variables for tracking recommendation and calculation flows
in_recommendation_mode = False
recommendation_state = {}
calculation_state = {}
in_calculation_mode = False

# Dictionary for mapping language codes to their names
language_names = {
    'en': 'English',
    'hi': 'Hindi',
    'gu': 'Gujarati'
}

def translate_text(text, source_lang, target_lang):
    """Translate text using Gemini."""
    if source_lang == target_lang:
        return text
    
    try:
        prompt = f"Translate the following text from {language_names.get(source_lang, source_lang)} to {language_names.get(target_lang, target_lang)}. Maintain the same tone and meaning. Here's the text: {text}"
        
        response = translation_model.generate_content(prompt)
        translated_text = response.text
        
        # Clean up any markdown formatting that might be in the response
        translated_text = translated_text.replace('```', '').strip()
        
        return translated_text
    
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def is_calculation_query(query):
    """Check if a query is calculation-based."""
    # Patterns for calculation queries
    sip_patterns = [
        r'sip', r'systematic investment plan', r'monthly investment',
        r'calculate (my )?returns', r'investment calculator', r'calculate interest'
    ]
    
    # Check for fund details queries starting with @
    fund_pattern = r'@([a-zA-Z0-9\s]+)'
    
    # Check if any SIP pattern matches
    for pattern in sip_patterns:
        if re.search(pattern, query.lower()):
            return "sip"
    
    # Check if the fund pattern matches
    if re.search(fund_pattern, query):
        return "fund"
    
    return None


def extract_fund_name(query):
    """Extract fund name from a query containing @."""
    match = re.search(r'@([a-zA-Z0-9\s\-]+)', query)
    if match:
        return match.group(1).strip()
    return None


def extract_sip_parameters(query, model):
    """Extract SIP calculation parameters from the query using Gemini."""
    try:
        prompt = """
        Extract SIP (Systematic Investment Plan) calculation parameters from the following query.
        Return a JSON with these keys:
        - monthly_investment: Monthly investment amount in numbers only
        - interest_rate: Annual interest rate as a percentage (number only)
        - time_period: Investment duration in years (number only)
        - lump_sum: Any initial lump sum amount (number only, 0 if not specified)
        
        If any parameter is missing, set its value to null.
        
        Query: {query}
        """
        
        response = model.generate_content(prompt.format(query=query))
        
        # Process the response to extract the JSON
        import json
        # Clean up the response to get just the JSON part
        response_text = response.text
        # Remove markdown code block markers if present
        response_text = response_text.replace('```json', '').replace('```', '')
        
        # Parse the JSON
        params = json.loads(response_text)
        return params
    
    except Exception as e:
        print(f"Error extracting SIP parameters: {e}")
        return {
            "monthly_investment": None,
            "interest_rate": None,
            "time_period": None,
            "lump_sum": 0
        }

def calculate_sip(monthly_investment, interest_rate, time_period, lump_sum=0):
    """
    Calculate SIP returns.
    
    Parameters:
    - monthly_investment: Monthly investment amount
    - interest_rate: Annual interest rate (as a percentage)
    - time_period: Investment duration in years
    - lump_sum: Initial lump sum investment (optional)
    
    Returns:
    - Dictionary with calculation results
    """
    if None in [monthly_investment, interest_rate, time_period]:
        return {"error": "Missing required parameters"}
    
    try:
        # Convert inputs to appropriate types
        monthly_investment = float(monthly_investment)
        interest_rate = float(interest_rate)
        time_period = float(time_period)
        lump_sum = float(lump_sum) if lump_sum is not None else 0
        
        # Convert annual interest rate to monthly
        monthly_rate = interest_rate / 12 / 100
        
        # Calculate number of months
        months = int(time_period * 12)
        
        # Calculate future value of SIP
        sip_future_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        
        # Calculate future value of lump sum
        lump_sum_future_value = lump_sum * ((1 + interest_rate/100) ** time_period)
        
        # Total future value
        total_future_value = sip_future_value + lump_sum_future_value
        
        # Total investment
        total_investment = monthly_investment * months + lump_sum
        
        # Total returns
        total_returns = total_future_value - total_investment
        
        # Prepare result
        result = {
            "monthly_investment": monthly_investment,
            "interest_rate": interest_rate,
            "time_period": time_period,
            "lump_sum": lump_sum,
            "total_investment": round(total_investment, 2),
            "total_returns": round(total_returns, 2),
            "total_future_value": round(total_future_value, 2),
            "missing_parameters": []
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}

def get_missing_parameters(params):
    """Identify which parameters are missing for SIP calculation."""
    missing = []
    if params["monthly_investment"] is None:
        missing.append("monthly_investment")
    if params["interest_rate"] is None:
        missing.append("interest_rate")
    if params["time_period"] is None:
        missing.append("time_period")
    return missing

def format_calculation_result(result, translate_func=None, target_lang='en'):
    """Format calculation results into a readable response."""
    if "error" in result:
        response = f"Error in calculation: {result['error']}"
        return translate_func(response, 'en', target_lang) if translate_func else response
    
    response = f"""
Based on your inputs:
- Monthly Investment: ₹{result['monthly_investment']:,.2f}
- Annual Interest Rate: {result['interest_rate']}%
- Time Period: {result['time_period']} years
- Initial Lump Sum: ₹{result['lump_sum']:,.2f}

Your SIP calculation results:
- Total Amount Invested: ₹{result['total_investment']:,.2f}
- Total Returns: ₹{result['total_returns']:,.2f}
- Maturity Value: ₹{result['total_future_value']:,.2f}

This calculation assumes a constant interest rate throughout the investment period.
"""
    return translate_func(response, 'en', target_lang) if translate_func else response

def update_sip_parameters(current_state, query, model):
    """Update SIP parameters based on user's response."""
    try:
        # Extract the value from the user's response
        prompt = f"""
        The user provided this response for a missing SIP parameter: "{query}"
        Extract just the numerical value from this response.
        Return only the number, with no additional text.
        """
        
        response = model.generate_content(prompt)
        extracted_value = float(response.text.strip())
        
        # Determine which parameter to update based on the missing parameters
        if "monthly_investment" in current_state.get("missing", []):
            current_state["params"]["monthly_investment"] = extracted_value
            current_state["missing"].remove("monthly_investment")
        elif "interest_rate" in current_state.get("missing", []):
            current_state["params"]["interest_rate"] = extracted_value
            current_state["missing"].remove("interest_rate")
        elif "time_period" in current_state.get("missing", []):
            current_state["params"]["time_period"] = extracted_value
            current_state["missing"].remove("time_period")
        
        return current_state
        
    except Exception as e:
        print(f"Error updating SIP parameters: {e}")
        return current_state

def analyze_fund_data(nav_data, question, fund_name, model):
    """
    Analyze fund NAV data based on user's question using Gemini.
    
    Parameters:
    - nav_data: List of NAV data points with dates
    - question: User's question about the fund
    - fund_name: Name of the fund
    - model: Gemini model instance
    
    Returns:
    - Analysis as a string
    """
    try:
        # Prepare data for analysis
        nav_data_sorted = sorted(nav_data, key=lambda x: datetime.strptime(x.get('date', ''), '%d-%m-%Y'))
        
        # Get latest and oldest NAV records
        latest_nav = nav_data_sorted[0]
        oldest_nav = nav_data_sorted[-1]
        
        # Calculate duration between dates in years
        latest_date = datetime.strptime(latest_nav.get('date', ''), '%d-%m-%Y')
        oldest_date = datetime.strptime(oldest_nav.get('date', ''), '%d-%m-%Y')
        days_diff = (latest_date - oldest_date).days
        years = days_diff / 365.25
        
        # Calculate growth metrics
        latest_nav_value = float(latest_nav.get('nav', 0))
        oldest_nav_value = float(oldest_nav.get('nav', 0))
        
        absolute_growth = ((latest_nav_value - oldest_nav_value) / oldest_nav_value) * 100
        annualized_growth = (((latest_nav_value / oldest_nav_value) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Calculate volatility (standard deviation of returns)
        daily_returns = []
        for i in range(1, min(100, len(nav_data_sorted))):  # Limit to 100 data points for efficiency
            current_nav = float(nav_data_sorted[i-1].get('nav', 0))
            previous_nav = float(nav_data_sorted[i].get('nav', 0))
            if previous_nav > 0:
                daily_return = (current_nav - previous_nav) / previous_nav
                daily_returns.append(daily_return)
        
        volatility = np.std(daily_returns) * np.sqrt(252) * 100 if daily_returns else 0  # Annualized volatility
        
        # Calculate 1-month, 3-month, 6-month, 1-year returns if data is available
        periods = {
            "1_month": 30,
            "3_month": 90,
            "6_month": 180,
            "1_year": 365
        }
        
        period_returns = {}
        for period_name, days in periods.items():
            if days_diff >= days:
                # Find the closest NAV to the period
                target_date = latest_date - timedelta(days=days)
                closest_nav = nav_data_sorted[0]
                min_diff = float('inf')
                
                for nav in nav_data_sorted:
                    nav_date = datetime.strptime(nav.get('date', ''), '%d-%m-%Y')
                    diff = abs((nav_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_nav = nav
                
                period_nav_value = float(closest_nav.get('nav', 0))
                period_return = ((latest_nav_value - period_nav_value) / period_nav_value) * 100
                period_returns[period_name] = period_return
        
        # Format period returns properly
        one_month = f"{period_returns.get('1_month'):.2f}%" if '1_month' in period_returns else 'Not available'
        three_month = f"{period_returns.get('3_month'):.2f}%" if '3_month' in period_returns else 'Not available'
        six_month = f"{period_returns.get('6_month'):.2f}%" if '6_month' in period_returns else 'Not available'
        one_year = f"{period_returns.get('1_year'):.2f}%" if '1_year' in period_returns else 'Not available'
        
        # Create a prompt that analyzes the fund data based on the user's question
        prompt = f"""
        As a financial advisor, analyze this mutual fund data and answer the following question:
        "{question}"
        
        Fund Information:
        - Fund Name: {fund_name}
        - Latest NAV: ₹{latest_nav_value} (as of {latest_nav.get('date')})
        - Initial NAV in dataset: ₹{oldest_nav_value} (as of {oldest_nav.get('date')})
        - Time Period: {years:.2f} years
        - Total Growth: {absolute_growth:.2f}%
        - Annualized Growth Rate: {annualized_growth:.2f}%
        - Volatility: {volatility:.2f}%
        
        Period Returns:
        - 1 Month: {one_month}
        - 3 Months: {three_month}
        - 6 Months: {six_month}
        - 1 Year: {one_year}
        
        Recent NAV Data (Last 5 entries):
        {json.dumps(nav_data_sorted[:5], indent=2)}
        
        Based on this data, provide:
        1. A direct answer to the user's question about the fund
        2. Additional relevant insights about the fund's performance
        3. A brief conclusion about the fund's performance relative to typical market expectations
        
        Keep your response concise, informative, and focused on the data provided.
        """
        
        # Generate analysis using Gemini
        response = model.generate_content(prompt)
        analysis = response.text
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing fund data: {e}")
        return f"I encountered an error while analyzing the fund data: {str(e)}. Please try again with a different question or fund."


def handle_calculation_query(query, language):
    """Handle calculation-based queries or fund analysis queries."""
    global calculation_state, in_calculation_mode

    # Translate query to English if needed
    if language != 'en':
        query_for_processing = translate_text(query, language, 'en')
    else:
        query_for_processing = query

    # Determine the type of calculation query
    calc_type = is_calculation_query(query_for_processing)

    # If we're already in calculation mode, handle the continuation of the flow
    if in_calculation_mode and "missing" in calculation_state and calculation_state["missing"]:
        calculation_state = update_sip_parameters(calculation_state, query_for_processing, calculation_model)

        if calculation_state["missing"]:
            missing_param_names = {
                "monthly_investment": "monthly investment amount",
                "interest_rate": "annual interest rate (as a percentage)",
                "time_period": "investment duration in years"
            }

            next_param = calculation_state["missing"][0]
            response = f"Please provide the {missing_param_names[next_param]}:"
            if language != 'en':
                response = translate_text(response, 'en', language)
            return jsonify({"response": response})
        else:
            params = calculation_state["params"]
            result = calculate_sip(
                params["monthly_investment"],
                params["interest_rate"],
                params["time_period"],
                params["lump_sum"]
            )

            response = format_calculation_result(result, translate_text, language)

            in_calculation_mode = False
            calculation_state = {}

            return jsonify({"response": response})

    # Handle new calculation queries
    if calc_type == "sip":
        in_calculation_mode = True
        params = extract_sip_parameters(query_for_processing, calculation_model)
        missing = get_missing_parameters(params)

        if missing:
            calculation_state = {
                "params": params,
                "missing": missing
            }

            missing_param_names = {
                "monthly_investment": "monthly investment amount",
                "interest_rate": "annual interest rate (as a percentage)",
                "time_period": "investment duration in years"
            }

            first_missing = missing[0]
            response = f"To calculate your SIP returns, please provide the {missing_param_names[first_missing]}:"
            if language != 'en':
                response = translate_text(response, 'en', language)
            return jsonify({"response": response})

        result = calculate_sip(
            params["monthly_investment"],
            params["interest_rate"],
            params["time_period"],
            params["lump_sum"]
        )

        response = format_calculation_result(result, translate_text, language)
        return jsonify({"response": response})

    elif calc_type == "fund":
        fund_name = extract_fund_name(query_for_processing)

        try:
            api_url = "https://api.mfapi.in/mf"
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            maps = {scheme.get("schemeName"): scheme.get("schemeCode") for scheme in data}
            all_fund_names = list(maps.keys())

            # Fuzzy match the fund name
            close_matches = difflib.get_close_matches(fund_name, all_fund_names, n=1, cutoff=0.6)

            if not close_matches:
                response_text = f"Fund name '{fund_name}' not found."
                if language != 'en':
                    response_text = translate_text(response_text, 'en', language)
                return jsonify({"response": response_text})

            best_match_name = close_matches[0]
            code = maps[best_match_name]

            past_url = f"https://api.mfapi.in/mf/{code}"
            past_response = requests.get(past_url)
            past_response.raise_for_status()
            past_data = past_response.json()

            if "data" not in past_data:
                response_text = f"No historical data available for fund '{best_match_name}'."
                if language != 'en':
                    response_text = translate_text(response_text, 'en', language)
                return jsonify({"response": response_text})

            question = query_for_processing.replace(f"@{fund_name}", "").strip()
            if not question:
                question = f"Provide an analysis of {best_match_name} mutual fund's performance"

            analysis = analyze_fund_data(past_data["data"], question, best_match_name, calculation_model)

            if language != 'en':
                analysis = translate_text(analysis, 'en', language)

            return jsonify({"response": analysis})

        except Exception as e:
            error_msg = f"Error analyzing fund {fund_name}: {str(e)}"
            if language != 'en':
                error_msg = translate_text(error_msg, 'en', language)
            return jsonify({"error": error_msg}), 500

    return None

    

def is_recommendation_request(query: str) -> bool:
    """Check if the query is asking for investment recommendations."""
    recommendation_keywords = [
        "recommend", "suggestion", "advice", "advise", "invest",
        "portfolio", "allocation", "strategy", "plan", "what should i invest",
        "help me invest", "investment options"
    ]
    return any(keyword in query.lower() for keyword in recommendation_keywords)

def handle_recommendation_logic(query: str, lang: str) -> str:
    """Handle the recommendation dialogue flow with translation support."""
    global recommendation_state

    # First translate the query to English for processing
    if lang != 'en':
        query_in_english = translate_text(query, lang, 'en')
    else:
        query_in_english = query

    response_in_english = ""
    
    if 'age' not in recommendation_state:
        age_match = re.search(r'\b(\d{1,2})\b(?:\s*(?:years?(?:\s*old)?)?)?', query_in_english.lower())
        if age_match:
            age = int(age_match.group(1))
            if 18 <= age <= 100:
                recommendation_state['age'] = age
                response_in_english = "What is your annual income (in dollars)?"
        else:
            response_in_english = "Please tell me your age (between 18-100):"

    elif 'income' not in recommendation_state:
        income_match = re.search(r'\b(\d+)(?:k)?\b', query_in_english.lower())
        if income_match:
            income_str = income_match.group(1)
            income = float(income_str)
            if 'k' in query_in_english.lower():
                income *= 1000
            recommendation_state['income'] = income
            response_in_english = "On a scale of 1-5, what's your risk tolerance? (1 being very conservative, 5 being very aggressive)"
        else:
            response_in_english = "Please specify your annual income (e.g., 50000 or 50k):"

    elif 'risk' not in recommendation_state:
        risk_match = re.search(r'[1-5]', query_in_english)
        if risk_match:
            risk = int(risk_match.group())
            recommendation_state['risk'] = risk
            # Generate recommendation
            response_in_english = get_recommendation(
                recommendation_state['age'],
                recommendation_state['income'],
                risk
            )
            # Clear the state after giving the recommendation
            recommendation_state = {}
            global in_recommendation_mode
            in_recommendation_mode = False
        else:
            response_in_english = "Please specify your risk tolerance (1-5):"
    
    else:
        response_in_english = "I apologize, but something went wrong. Let's start over. What's your age?"

    # Translate response back to target language if needed
    if lang != 'en':
        return translate_text(response_in_english, 'en', lang)
    else:
        return response_in_english

def chunk_response(text, chunk_size=300):
    """Split a long response into chunks of max `chunk_size` characters."""
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        if sum(len(w) for w in chunk) + len(word) + len(chunk) > chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
        chunk.append(word)
    
    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

@app.route("/chat", methods=["POST"])
def chat():
    global user_sessions, in_recommendation_mode, in_calculation_mode

    try:
        data = request.get_json()
        user_id = data.get("user_id", "default_user")  # Track user sessions
        user_message = data.get("query")
        language = data.get("language", "en")  # Get selected language, default to English
        
        valid_languages = ['en', 'hi', 'gu']
        if language not in valid_languages:
            return jsonify({"error": "Invalid language selected. Please choose English, Hindi, or Gujarati."}), 400

        if not user_message:
            return jsonify({"error": "'query' field is missing"}), 400

        # Check if user has pending message chunks
        if user_id in user_sessions and user_sessions[user_id]:
            remaining_chunks = user_sessions[user_id].pop(0)  # Get next chunk
            
            # Translate remaining chunk if needed
            if language != 'en':
                remaining_chunks = translate_text(remaining_chunks, 'en', language)
                
            return jsonify({"response": remaining_chunks})

        # Handle recommendation flow
        if is_recommendation_request(user_message) and not in_calculation_mode:
            in_recommendation_mode = True
            response = handle_recommendation_logic(user_message, language)
            return jsonify({"response": response})
        elif in_recommendation_mode:
            response = handle_recommendation_logic(user_message, language)
            return jsonify({"response": response})

        # Handle calculation/analysis flow (SIP or fund analysis)
        calculation_response = handle_calculation_query(user_message, language)
        if calculation_response:
            return calculation_response

        # If not a special query, process with RAG assistant
        
        # Translate user message to English if not in English
        if language != 'en':
            query_for_processing = translate_text(user_message, language, 'en')
        else:
            query_for_processing = user_message

        # Create message for Pinecone Assistant
        msg = Message(content=query_for_processing)
        response = assistant.chat(messages=[msg], stream=False)

        if 'message' not in response:
            return jsonify({"error": "Invalid response from assistant"}), 500

        # Get the full response in English
        full_response_english = response["message"]["content"]
        
        # Translate back to user's language if needed
        if language != 'en':
            full_response = translate_text(full_response_english, 'en', language)
        else:
            full_response = full_response_english

        # Split response into multiple messages
        response_chunks = chunk_response(full_response)

        # Store remaining chunks in session
        if len(response_chunks) > 1:
            user_sessions[user_id] = response_chunks[1:]

        return jsonify({"response": response_chunks[0]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/<fundname>', methods=['GET'])
def get_details(fundname):
    """Fetch past and present details for a specific mutual fund."""
    api_url = "https://api.mfapi.in/mf"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Find the code for the given fund name
        maps = {scheme.get("schemeName"): scheme.get("schemeCode") for scheme in data}
        code = maps.get(fundname)

        if not code:
            return jsonify({"error": f"Fund name '{fundname}' not found."}), 404

        # Fetch past data for the given fund code
        past_url = f"https://api.mfapi.in/mf/{code}/latest"
        past_response = requests.get(past_url)
        past_response.raise_for_status()
        past_data = past_response.json()


        if "data" not in past_data:
            return jsonify({"error": f"No historical data available for fund '{fundname}'."}), 404
        
        return jsonify(past_data)      

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch data for fund '{fundname}': {str(e)}"}), 500

@app.route('/schemes', methods=['GET'])
def get_names():
    """Fetch all mutual fund schemes and return scheme names."""
    api_url = "https://api.mfapi.in/mf"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Build a list of scheme names
        scheme_names = [scheme.get("schemeName") for scheme in data if scheme.get("schemeName")]
        
        return jsonify(scheme_names)  
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch mutual fund codes: {str(e)}"}), 500
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)