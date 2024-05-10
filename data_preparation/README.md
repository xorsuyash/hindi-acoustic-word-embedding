# Force Alignment 
Force alignment in audio data processing refers to the process of synchronizing audio segments with their corresponding text transcripts or labels, This task is crucial in 
generating dataset for **acoutic word embedding**. 

**This is the tool for force alignment of hindi audio 💪**

![Screenshot from 2024-04-28 16-47-25](https://github.com/xorsuyash/hindi-acoustic-word-embedding/assets/98162846/0076c28e-5266-4961-af75-3a1669a9de23)

# How to run the project 
* Create and activate either virtual enviroment or conda enviroment.
* Navigate to data preparation folder.

      cd data_preparation
* Install requirements

      pip install -r requirements.txt

**Code Usage 👨‍💻**

Audio file must be in .wav format and along with its correct hindi-transcript.
  ```python
      from alignment import force_align

      input_path="<audio_file_path>"
      transcript="transcript of the audio"
      output_folder="<path_of_output_folder>"

      force_align(input_path, transcript, output_folder)

