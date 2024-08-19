import pandas as pd

df = pd.read_excel('./perceptual_results/results_diffusion_with_wer.xlsx')

# Function to count words
def count_words(text):
    return len(text.split())

df['ratio_words'] = df.apply(lambda row: count_words(row['Response Corrected']) / count_words(row['Transcription']), axis=1)

print('MEAN RATIO WORDS')
print(df[df['audio_type'] == 'ljspeech']['ratio_words'].mean())
# Filter for groundtruth rows
groundtruth_df = df[df['audio_type'] == 'groundtruth']

# Calculate mean WER for each participant in groundtruth rows
mean_wer_per_participant = groundtruth_df.groupby('Participant Private ID')['wer'].mean()

# Identify participants with mean WER below 0.90
participants_above_threshold = mean_wer_per_participant[mean_wer_per_participant > 0.10].index

print('NEW NUMBER OF PARTICIPANTS')
num_participants_evicted = len(participants_above_threshold)
num_total_participants = len(df['Participant Private ID'].unique())
num_participants = num_total_participants - num_participants_evicted
print(f"{num_participants} / {num_total_participants}")
print(participants_above_threshold)

# df_margaux = df[df['Participant Private ID']==11515302]
# result_df_margaux = df_margaux.groupby('audio_type')['wer'].agg(['mean', 'std']).reset_index()
# print('MARGAUX')
# print(result_df_margaux)

# Filter out these participants from the original DataFrame
df_evicted = df[df['Participant Private ID'].isin(participants_above_threshold)]
df = df[~df['Participant Private ID'].isin(participants_above_threshold)]

result_df_evicted = df_evicted.groupby('audio_type')['wer'].agg(['mean', 'std']).reset_index()
result_df = df.groupby('audio_type')['wer'].agg(['mean', 'std']).reset_index()

# Save the results to a new Excel file
df_evicted[df_evicted['audio_type'] == 'groundtruth'].to_excel('./perceptual_results/evicted.xlsx', index=False)

print('RESULTS')
print(result_df)
print(result_df_evicted)