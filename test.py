from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(api_key="AIzaSyD7DB3FvAlE5lS_5Ezt91WhOX_EgqmixjM" , model="gemini-2.5-flash")
a = llm.invoke("WHAT IS THIS")
print(a)

