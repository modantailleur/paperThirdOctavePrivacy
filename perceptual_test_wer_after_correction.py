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

# data = pd.read_csv('./perceptual_results/data_exp_183159-v14_task-or5j.csv')
data = pd.read_excel('./perceptual_results/results_diffusion_with_manual_corrections.xlsx')

wermetric = WERMetric()

data['wer'] = data.apply(wermetric.calculate_wer_with_corrections, axis=1)

# Print the entire dataframe
data.to_excel('./perceptual_results/results_diffusion_with_wer.xlsx', index=False)

