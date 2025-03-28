from google import genai
from config.settings import GEMINI_API_KEY  # Import API key from config

class WasteManagementChatbot:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)  # Initialize GenAI client
        # List of greeting patterns to check
        self.greetings = ["hi", "hello", "hey", "greetings", "howdy", "hi there", "hello there"]
    
    def generate_waste_advice(self, user_input):
        """Generates waste management advice using Gemini."""
        # Check if input is just a greeting
        if self._is_greeting(user_input):
            return {
                "response": "Hello! I'm your RegenBot. How may I help you today with waste management questions? Ask me about how to recycle or dispose of specific items!"
            }
            
        prompt = (
            f"Based on the following waste item or question: '{user_input}', provide practical waste management advice "
            f"with the following clearly marked sections in markdown format:\n"
            f"1. **Classification:** (recyclable, compostable, hazardous, landfill, etc.)\n"
            f"2. **Disposal Instructions:** (step-by-step guidance)\n"
            f"3. **Environmental Impact:** (brief explanation)\n"
            f"4. **Alternatives:** (eco-friendly alternatives if applicable)\n"
            f"Be concise and direct. Focus on practical advice rather than lengthy explanations. "
            f"If the query isn't about a specific waste item, provide relevant waste management information."
        )
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt])
            advice_text = response.text
            print(advice_text) # for debugging
            
            # Add disclaimer
            advice_text += "\n\n*This advice was generated by an AI. Local regulations may vary, so check with your municipality for specific guidelines.*"
            
            return {
                "response": advice_text
            }
            
        except Exception as e:
            error_message = f"Error generating waste management advice: {e}"
            print(error_message)
            return {
                "response": "I'm having trouble processing your request right now. Please try again with a more specific question about waste management."
            }
    
    def _is_greeting(self, text):
        """Check if the input is just a simple greeting."""
        text = text.lower().strip()
        
        # Check if the text contains only a greeting
        for greeting in self.greetings:
            # Check if it's exactly the greeting or the greeting + "regenbot"
            if text == greeting or text.startswith(greeting + " ") or "regenbot" in text:
                return True
        
        # Check for very short inputs (less than 4 words) that may be greetings
        if len(text.split()) < 4 and ("hi" in text or "hey" in text or "hello" in text):
            return True
            
        return False

# Format response data into a readable message
def format_response_data(data):
    # Special case for greetings
    if data["classification"] == "Greeting":
        return data["disposal_instructions"]
    
    # Normal formatting for waste management advice
    return f"**Classification:** {data['classification']}\n\n" + \
           f"**Disposal Instructions:** {data['disposal_instructions']}\n\n" + \
           f"**Environmental Impact:** {data['environmental_impact']}\n\n" + \
           f"**Eco-Friendly Alternatives:** {data['eco_alternatives']}\n\n" + \
           f"*{data['additional_notes']}*"