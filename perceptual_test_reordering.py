import pandas as pd
import jiwer
from transformers import pipeline

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class FilenameMatcher():
    def __init__(self, transcription):
        self.transcription = transcription
        self.transcription_dict = transcription.set_index('Filename')['Transcription'].to_dict()
    def find_transcription(self, screen):
        for filename, transcription in self.transcription_dict.items():
            if filename in screen:
                return transcription
        return ''

# Helper function to determine audio type
def determine_audio_type(screen):
    if 'normalized_LIBRI+' in screen:
        return 'librispeech'
    elif 'normalized_' in screen:
        return 'ljspeech'
    else:
        return 'groundtruth'

# Helper class to calculate WER
class WERMetric():
    def __init__(self):
        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
    # Helper function to calculate WER
    def calculate_wer_with_corrections(self, row):
        transcript_target = row['Transcription']
        transcript_test = row['Response Corrected']
        if pd.notna(transcript_target) and pd.notna(transcript_test):
            wer = jiwer.wer(
                            transcript_target,
                            transcript_test,
                            truth_transform=self.transforms,
                            hypothesis_transform=self.transforms,
                        )
            wer = 1 if wer > 1 else wer
            return wer
        else:
            return None
        
    def calculate_wer(self, row):
        transcript_target = row['Transcription']
        transcript_test = row['Response']
        if pd.notna(transcript_target) and pd.notna(transcript_test):
            wer = jiwer.wer(
                            transcript_target,
                            transcript_test,
                            truth_transform=self.transforms,
                            hypothesis_transform=self.transforms,
                        )
            wer = 1 if wer > 1 else wer
            return wer
        else:
            return None
        
# Helper class to correct spelling
class SpellingCorrection():
    def __init__(self, n_samples=None):
        try:
            self.fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")
        except Exception as e:
            print(f"Error initializing spelling correction pipeline: {e}")
            self.fix_spelling = None
        self._correction_idx = 0
        self.n_samples = n_samples

    def correct_spelling(self, text):
        if self.n_samples is not None:
            print(f"PROGRESSION: {self._correction_idx} / {self.n_samples}", end='\r')
        else:
            print(f"PROGRESSION: {self._correction_idx}", end='\r')
        self._correction_idx += 1
        if self.fix_spelling is None:
            return text
        if pd.isna(text) or not text.strip():
            return ''
        try:
            corrected_text = self.fix_spelling(text, max_length=2048)[0]['generated_text']
        except Exception as e:
            print(f"Error during spelling correction: {e}")
            corrected_text = text
        return corrected_text

# data = pd.read_csv('./perceptual_results/data_exp_183159-v14_task-or5j.csv')
data = pd.read_csv('./perceptual_results/data_exp_183159-v14_task-or5j.csv')
transcription = pd.read_excel('./perceptual_results/transcription.xlsx')
# keep columns Response, Participant Private ID, and Spreadsheet: screen

data = data[['Response', 'Participant Private ID', 'Spreadsheet: screen']]
data = data.dropna()
data = data[~data['Response'].isin(['audio started', 'audio finished', 
                                    'audio paused', 'audio seek started', 'audio seek backwards', 'audio seek forwards'])]
data = data[~data['Response'].str.contains('xxx')]

# Create a dictionary from the transcription DataFrame for faster lookups
transcription_dict = transcription.set_index('Filename')['Transcription'].to_dict()

wermetric = WERMetric()
spelling_correction = SpellingCorrection(n_samples=len(data))
filename_matcher = FilenameMatcher(transcription)

# Apply the helper function to the 'Spreadsheet: screen' column
data['audio_type'] = data['Spreadsheet: screen'].apply(determine_audio_type)
data['Transcription'] = data['Spreadsheet: screen'].apply(filename_matcher.find_transcription)
# data['Response Corrected'] = data['Response'].apply(spelling_correction.correct_spelling)
# data['wer'] = data.apply(wermetric.calculate_wer, axis=1)
# data['wer_corrected'] = data.apply(wermetric.calculate_wer_with_corrections, axis=1)

# Reorder the columns
data = data[['Participant Private ID', 'audio_type', 'Spreadsheet: screen', 'Response', 'Transcription']]

# Print the entire dataframe
data.to_excel('./perceptual_results/results_diffusion.xlsx', index=False)
