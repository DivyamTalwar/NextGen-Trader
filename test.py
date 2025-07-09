from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(api_key="AIzaSyD5-aIJq8Y_JUCnc4KCCcRXEuTCR2bFuZU" , model="gemini-2.5-flash")
a = llm.invoke("WHAT IS THIS")
print(a)

