import random
import string
import numpy as np

def generate_captcha():
    # Generate a random CAPTCHA with a combination of letters and numbers
    captcha_length = 6
    captcha_characters = string.ascii_letters + string.digits
    captcha = ''.join(random.choice(captcha_characters) for _ in range(captcha_length))
    return captcha

def display_captcha(captcha):
    # Display the CAPTCHA to the user
    print("CAPTCHA:", captcha)

def verify_captcha(input_captcha, generated_captcha):
    # Verify if the user input matches the generated CAPTCHA
    return input_captcha.lower() == generated_captcha.lower()

# Main verification loop
while True:
    # Generate a new CAPTCHA
    generated_captcha = generate_captcha()

    # Display the CAPTCHA
    display_captcha(generated_captcha)

    # Get user input
    user_input = input("Enter the CAPTCHA: ")

    # Verify the input
    if verify_captcha(user_input, generated_captcha):
        print("Verification Successful! Access Granted.")
        break
    else:
        print("Verification Failed. Please try again.")
