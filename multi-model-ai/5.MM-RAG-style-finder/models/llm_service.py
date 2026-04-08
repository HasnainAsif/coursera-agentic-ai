"""
Service for interacting with the OpenAI Vision Instruct model.
"""

import logging
import os
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIVisionService:
    """
    Provides methods to interact with the OpenAI Vision Instruct model.
    """
    
    def __init__(self, model_id, temperature=0.2, api_key=None, max_tokens=2000):
        """
        Initialize the service with the specified model and parameters.
        
        Args:
            model_id (str): Unique identifier for the model
            temperature (float): Controls randomness in generation
            api_key (str, optional): API key for authentication
            max_tokens (int): Maximum tokens in the response
        """
        # Initialize OpenAI client
        self.client = OpenAI()
        self.model_id = model_id
    
    def generate_response(self, encoded_image, prompt):
        """
        Generate a response from the model based on an image and prompt.
        
        Args:
            encoded_image (str): Base64-encoded image string
            prompt (str): Text prompt to guide the model's response
            
        Returns:
            str: Model's response
        """
        try:
            logger.info("Sending request to LLM with prompt length: %d", len(prompt))
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + encoded_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            content = response.choices[0].message.content
            
            logger.info("Received response with length: %d", len(content))
            
            if len(content) >= 7900:
                logger.warning("Response may be truncated (length: %d)", len(content))
            
            return content
            
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"
    
    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
        """
        Generate a fashion-specific response using role-based prompts.
        """
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} (${row['Price']}): {row['Link']}"
            items_list.append(item_str)
        
        items_description = "\n".join([f"- {item}" for item in items_list])

        if similarity_score >= threshold:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"ITEM DETAILS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
                "2. Categorize the overall style (business, casual, etc.)\n"
                "3. Include the ITEM DETAILS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        else:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"SIMILAR ITEMS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Note these are similar but not exact items\n"
                "2. Identify clothing elements objectively (colors, patterns, materials)\n" 
                "3. Include the SIMILAR ITEMS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        
        response = self.generate_response(user_image_base64, assistant_prompt)
        
        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response = f"# Fashion Analysis\n\nThis outfit features a collection of carefully coordinated pieces.\n\n{section_header}\n{items_description}"
        
        elif "ITEM DETAILS:" not in response and "SIMILAR ITEMS:" not in response:
            logger.info("Item details section missing from response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response += f"\n\n{section_header}\n{items_description}"
        
        return response
