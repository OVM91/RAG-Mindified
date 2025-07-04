# RAG-Mindified

Step-By-Step:
1) data_processing.py -
Takes RAW-data which consists of multiple conversations (each conversation having its own metadata) and transforms it into one transcript with relevant metadata.
During this process we also clean some redundant system-messages and add the amount of duplicates as metadata instead (in case customer find the info relevant).

2) main.py -
Loads the transformed data (output data from the previous process), iterate each message, its metadata and a prompt to a LLM for extracting
requested information from each message and structure them into a JSON schema as output --> Saves file.
We also use pydantic to validate the data types for the structured output of the LLM (not mandatory for now, prompt needs more precise input in regards to data types for consistency).
In case there is an error in the structured output of the llm we save the necessary info in a file called failed 'failed_conversations.json for debugging'

3) evaluation_rag.py - 
Evaluate the structured output of the LLM against an 'answer sheet/grounded truth' with a LLM as a judge, keeps score and prints out the total
score and accuracy --> Saves file with the feedback from the LLM-as-a-judge.
LLM-as-a-judge is not that reliable, for more robust solution in regards to evaluation its probably better to hard code the eval for some of the fields.

Övrigt:
Bl.a. Lägga till dynamiska filnamn så att man slipper själv ändra dessa varje gång man testar olika llm_modeller.
Mer robust error-handling och logging.
