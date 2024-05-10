import sys
from alignment import force_align

def print_help():
    print("""
Usage: python cli_tool.py <input_audio_path> <transcript> <output_folder_path>

Description:
This CLI tool processes audio files and transcripts and saves the output in a specified folder.

Arguments:
- input_audio_path: Path to the input audio file.
- transcript: Text transcript of the audio file.
- output_folder_path: Path to the folder where the output will be saved.
""")

def main():

    if "--help" in sys.argv:
        print_help()
        return
    
    if len(sys.argv) != 4:
        print("Error: Incorrect number of arguments.")
        print_help()
        return
    
    input_audio_path = sys.argv[1]
    transcript = sys.argv[2]
    output_folder_path = sys.argv[3]

    force_align(input_audio_path,transcript,output_folder_path)

    

if __name__ == "__main__":
    main()