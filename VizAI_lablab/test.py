import os
from crewai import Agent, Task, Crew, Process
from openai import OpenAI
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM

# Upstage API configuration
UPSTAGE_API_KEY = ""
BASE_URL = "https://api.upstage.ai/v1/solar"

# Custom LLM class for Upstage API
class UpstageCustomLLM(LLM):
	client: OpenAI = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.client = OpenAI(
			api_key=UPSTAGE_API_KEY,
			base_url=BASE_URL
		)

	@property
	def _llm_type(self) -> str:
		return "upstage_custom"

	def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
		messages = [
			{"role": "system", "content": "You are a recipe generator. Create recipes using ONLY the ingredients provided. Do not add or suggest any additional ingredients."},
			{"role": "user", "content": prompt}
		]

		response = self.client.chat.completions.create(
			model="solar-pro",
			messages=messages,
			stream=False
		)

		return response.choices[0].message.content.replace("Â", "")  # Remove weird symbols

	@property
	def _identifying_params(self) -> Mapping[str, Any]:
		return {"name": "UpstageCustomLLM"}

# Create a Crew AI agent for recipe generation
recipe_agent = Agent(
	role="Chef",
	goal="Create a delicious recipe using given ingredients",
	backstory="I am an experienced chef with knowledge of various cuisines.",
	allow_delegation=False,
	verbose=True,
	llm=UpstageCustomLLM()
)

# Create a Crew with our recipe agent
crew = Crew(
	agents=[recipe_agent],
	tasks=[
		Task(
			description="Generate a recipe using the given ingredients",
			agent=recipe_agent,
			expected_output="A complete recipe with title and numbered steps."
		)
	]
)

def generate_recipe(ingredients):
	print("\nIngredients:", ", ".join(ingredients))
	print("\nGenerating Recipe:")
	
	# Use Crew AI to generate and refine the recipe
	result = crew.kickoff(inputs={
		"ingredients": ingredients,
		"instructions": f"Create a recipe using ONLY these ingredients: {', '.join(ingredients)}. The recipe must not use any ingredients not listed. Provide a title for the dish and list the steps in a numbered format."
	})
	
	# Validate the recipe
	while any(ingredient.lower() not in result.lower() for ingredient in ingredients):
		print("Recipe didn't use all ingredients. Regenerating...")
		result = crew.kickoff(inputs={
			"ingredients": ingredients,
			"instructions": f"The previous recipe was incorrect. Create a new recipe using ALL and ONLY these ingredients: {', '.join(ingredients)}. Do not add any other ingredients. Provide a title for the dish and list the steps in a numbered format."
		})
	
	return result

# Example usage
if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		ingredients = sys.argv[1].split(',')
	else:
		ingredients = ["eggs", "pasta", "cheese", "bacon"]
	final_recipe = generate_recipe(ingredients)
	print("\nFinal Recipe:")
	print(final_recipe)
