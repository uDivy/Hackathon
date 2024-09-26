import streamlit as st
import subprocess

def get_recipe(ingredients):
    # Join ingredients with commas for passing to test.py
    ingredients_str = ','.join(ingredients)
    # Run test.py with the ingredients as an argument
    result = subprocess.run(['python', 'test.py', ingredients_str], capture_output=True, text=True)
    return result.stdout

st.title("Recipe Generator")

# Initialize session state for ingredients if it doesn't exist
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = []

# Input field for new ingredient
new_ingredient = st.text_input("Enter an ingredient:")

# Button to add the ingredient to the list
if st.button("Add Ingredient"):
    if new_ingredient and new_ingredient not in st.session_state.ingredients:
        st.session_state.ingredients.append(new_ingredient)
        st.success(f"Added {new_ingredient} to the list.")
    elif new_ingredient in st.session_state.ingredients:
        st.warning(f"{new_ingredient} is already in the list.")
    else:
        st.warning("Please enter an ingredient.")

# Display current list of ingredients
st.write("Current ingredients:", ", ".join(st.session_state.ingredients))

# Button to generate recipe
if st.button("Get Recipe"):
    if st.session_state.ingredients:
        recipe = get_recipe(st.session_state.ingredients)
        st.write("Recipe:")
        st.write(recipe)
    else:
        st.warning("Please add at least one ingredient.")

# Button to clear all ingredients
if st.button("Clear All Ingredients"):
    st.session_state.ingredients = []
    st.success("All ingredients cleared.")
