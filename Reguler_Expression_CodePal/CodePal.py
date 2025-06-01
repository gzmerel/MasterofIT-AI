# Import the 're' module to enable the use of regular expressions
import re
# Parameters:
#   message (str): A string representing the student's message.
# Returns:
#   str: A helpful response from CodePal, based on detected keywords or phrases.

# Define the chatbot response function using regular expressions
def codepal_response(message: str) -> str:
  #Convert message to lowercase to make matching easier.
  message = message.lower()
  #Debugging concerns
  if re.search(r"\b(debug|bug|stuck|issue|troubleshoot)\b",message):
    return "Debugging can be frustrating. Have you tried printing intermediate results?"
  #Code error or compile issue
  elif re.search(r"\b(compile|error|crash|crashing)\b",message):
    return "Let's take a deep breath and go step-by-step. What does the error message say?"
  #Understanding of concepts or confusion
  elif re.search(r"\b(loop|recursion|hard|confusing|topic)\b",message):
    return "That's a common hurdle. Would you like a resource that explains it visually?"
  #Asking about the bot
  elif re.search(r"(who are you|what are you|are you human)",message):
    return "I'm CodePal, your friendly code companion designed to support your learning journey."
  #Farewell
  elif re.search(r"\b(bye|exit|logout|gotta go|see you)\b",message):
    return "Happy coding! Remember, every bug is a step forward."
  #Anything different than keywords above
  else:
    return "Let’s tackle it together — what’s your next coding challenge?"
# Define the function to simulate a conversation
# This function runs a loop to simulate a conversation with CodePal.
def start_chat():
    print("You can start talking to the chatbot! Type 'bye' to end the conversation.\n")

    while True:
        # Get user input
        student_input = input("Student: ")
        # Generate response
        bot_response = codepal_response(student_input)
        # Print bot response
        print(f"CodePal: {bot_response}")

        # Exit if a farewell is detected
        if re.search(r"\b(bye|exit|logout|gotta go|see you)\b", student_input.lower()):
            break
# Run the conversation simulation function
# Start the chatbot interaction
start_chat()
