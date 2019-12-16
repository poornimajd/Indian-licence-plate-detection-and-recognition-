 OCR FOR NUMBER PLATE RECOGNITION

-- "objectx.py" consists the server side program.
   // set the nms_thresh and confidence if required. default - 0.1 for both. 
   Requirements (change the paths in objectx.py file)
   - Configuration file
   - Weights file in .weights format (torch usage)
   - obj.names file
   - plate.py
   
   returns a list of characters.

-- "pyclient.py" consists the server side program.
   // change path to image folder
   Requirements 
   - scripts folder
   - images folder
   
   returns a string.

-- "plate.py" consists the logic to read the number plate in the correct order.
   // Do not change anything in the file. Only change if a change required in the logic of the code.
