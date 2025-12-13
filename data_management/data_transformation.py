import pandas as pd
from scipy import stats
# # from google.colab import drive
import re
import numpy as np
import matplotlib.pyplot as plt
import math
# from os import pardir

debug = False

# String literal column names for transformed dataframe
part_id_col = "part_id"
stim_src_id_col = "stim_src_id"
stim_order_col = "stim_order"
has_bug_col = "has_bug"
condition1_col = "condition1"
condition2_col = "condition2"
seq_num_col = "seq_num"
duration_col = "duration"
exercise_col = "exercise"
ai_col = "ai"
tp_tn_col = "tp_tn"
zoom_self_col = "zoom_self"
defect_type_col = "defect_type"
file_path_col = "file_path"
method_loc_col = "method_loc"
file_loc_col = "file_loc"
cyclomatic_compl_col = "cyclomatic_compl"
halstead_vocabulary_col = "halstead_vocabulary"
halstead_length_col = "halstead_length"
halstead_volume_col = "halstead_volume"
halstead_difficulty_col = "halstead_difficulty"
halstead_effort_col = "halstead_effort"
halstead_time_seconds_col = "halstead_time_seconds"
halstead_estimated_bugs_col = "halstead_estimated_bugs"
method_name_col = "method_name"
project_name_col = "project_name"
# Task questions
summary_q_col = "summary_q"
summary_q_len_col = "summary_q_len"
contains_bug_q_col = "contains_bug_q"
bug_descr_q_col = "bug_desc_q"
bug_descr_q_len_col = "bug_desc_q_len"
# Post-task questions -- exercise
used_exercise_modif_col = "used_exercise_modif"
exercise_disrupt_concentration_col = "exercise_disrupt_concentration"
exercise_help_concentration_col = "exercise_help_concentration"
exercise_more_prepared_col = "exercise_more_prepared"
exercise_less_prepared_col = "exercise_less_prepared"
# Post-task questions -- AI
read_ai_col = "read_ai"
ai_deceptive_col = "ai_deceptive"
ai_suspicious_col = "ai_suspicious"
ai_confident_col = "ai_confident"
ai_dependable_col = "ai_dependable"
# Post-task questions -- experience
satisfied_ease_col = "satisfied_ease"
satisfied_time_col = "satisfied_time"
confidence_col = "confidence"
# Demographic questions
race_ethnic_col = "race/ethnic"
gender_col = "gender"
occupation_col = "occupation"
occupation_other_col = "occupation_7_TEXT"
um_year_col = "um_year"
prog_exp_1_to_10_col = "prog_exp_1_to_10"
describe_prog_context_col = "describe_prog_contex"
approx_writing_code_col = "approx_writing_code"
level_of_education_col = "level_of_education"
			# TODO race/ethnic _8_TEXT, gender _5_TEXT, occupation _7_TEXT, describe_prog_contex_9_TEXT, 
# Demographic questions -- exercise
days_vigorous_col = "days_vigorous"
minutes_vigorous_col = "minutes_vigorous"
days_moderate_col = "days_moderate"
minutes_moderate_col = "minutes_moderate"
days_walking_col = "days_walking"
minutes_walking_col = "minutes_walking"
hours_sitting_col = "hours_sitting"
# Demographic questions -- language
language_at_home_col = "language_at_home"
language_prefer_info_col = "language_prefer_info"
how_well_speak_engl_col = "how_well_speak_engl"
how_well_read_engl_col = "how_well_read_engl"
list_languages_1_col = "list_languages_1" 
list_languages_2_col = "list_languages_2" 
list_languages_3_col = "list_languages_3" 
list_languages_4_col = "list_languages_4" 
list_languages_5_col = "list_languages_5" 
list_lang_percents_1_col = "list_lang_percents_1"
list_lang_percents_2_col = "list_lang_percents_2"
list_lang_percents_3_col = "list_lang_percents_3"
list_lang_percents_4_col = "list_lang_percents_4"
list_lang_percents_5_col = "list_lang_percents_5"
# Qualitative annotation scores
readability_col = "readability"
completeness_col = "completeness"
conciseness_col = "conciseness"
accuracy_col = "accuracy"
correctly_identifies_bug_col = "correctly_identifies_bug"
reasoning_col = "reasoning"
themes_col = "themes"


# Constant strings
TP = "TP"
TN = "TN"
FP = "FP"
FN = "FN"
exercise_ai = "exercise_ai"
exercise_no_ai = "exercise_no_ai"
no_exercise_ai = "no_exercise_ai"
no_exercise_no_ai = "no_exercise_no_ai"

# Clean and buggy stimuli src ids
clean = [
   8,  6, 22,  4, 15,  7, 19,
  28, 12, 13, 25, 26, 32, 27,
  33, 29, 30
]
buggy = [
   1,  2,  3,  9,  5, 18, 10,
  11, 17, 20, 14, 16, 21, 23,
  24, 31, 34
]


def transform_dataframe(df):
	'''
	Transforms csv file into dataframe where each row is participant response: (part_id, git_src_id)
	Independent: part_id, git_src_id, ai, tp_tn, post_exercise, seq_num, was_supervised, code_exp, exer_exp
	Dependent: summary_score, found_bug, bug_descr_score, time
	'''

 	# Transform CSV file so that each participant id has 6 rows (one per task)
	new_rows = [] # Collect rows to create new, transformed dataframe

	for idx, row in df.iterrows():
		part_id = row["part_id"]

		if (not isinstance(part_id, str)):
			continue
		if (not re.match(r"^[a-zA-Z]{3}\d$", part_id)):
			continue
		if (re.match(r"^[a-z]{3}\d$", part_id)):
			# Convert lowercase to uppercase
			part_id = part_id.upper()

		stimuli_order_str = row["__js_stimuliOrder"] # E.g., ",33_TN,12_FP,9_TP,3,28,5,"

		stimuli_order = stimuli_order_str.split(",")[1:-1] # E.g., ['33_TN', '12_FP', '9_TP', '3', '28', '5']

		# For each stimulus, create row to add to new dataframe
		for i in range(len(stimuli_order)):
			stimulus = stimuli_order[i]

			new_row = { part_id_col : part_id }
			# # Special cases -- exclusion
			# if (new_row[part_id_col] == "NWJ7") and (i > 1): # 
			# 	continue
			# TODO -- should we exclude this participant? Outlier?
			# if (new_row[part_id_col] == "AMF1"):
			# 	continue

			stim_src_id = stimulus.split("_")[0] # E.g., '33'
			new_row[stim_src_id_col] = stim_src_id
			new_row[has_bug_col] = int(stim_src_id) in buggy
			new_row[stim_order_col] = stimuli_order_str

			condition1, condition2 = row["condition"], row["condition2"]
			new_row[condition1_col] = condition1
			new_row[condition2_col] = condition2
			new_row[seq_num_col] = i+1
			new_row[duration_col] = float( row[f"{stim_src_id}_timing_Page Submit"] )

			# If exercise is true
			if i < 3:
				new_row[exercise_col] = (condition1 == exercise_ai) or (condition1 == exercise_no_ai)
			else:
				new_row[exercise_col] = (condition2 == exercise_ai) or (condition2 == exercise_no_ai)

			# If AI is true
			if i < 3:
				new_row[ai_col] = (condition1 == exercise_ai) or (condition1 == no_exercise_ai)
			else:
				new_row[ai_col] = (condition2 == exercise_ai) or (condition2 == no_exercise_ai)
			if new_row[ai_col]:
				assert("_" in stimulus)
				new_row[tp_tn_col] = stimulus.split("_")[1] # E.g., 'TP'

			# Zoom-supervised or Self-paced format
			if "Zoom" in row["selected_format"]:
				new_row[zoom_self_col] = "zoom"
			else:
				new_row[zoom_self_col] = "self"
			
			# Task questions
			new_row[summary_q_col] = row[f"{stim_src_id}_q1"]
			if isinstance(new_row[summary_q_col], str):
				new_row[summary_q_len_col] = len(new_row[summary_q_col])
			new_row[contains_bug_q_col] = row[f"{stim_src_id}_q2"]
			new_row[bug_descr_q_col] = row[f"{stim_src_id}_q3"]
			if isinstance(new_row[bug_descr_q_col], str):
				new_row[bug_descr_q_len_col] = len(new_row[bug_descr_q_col])

			# Post-task questions
			if new_row[exercise_col]:
				new_row[used_exercise_modif_col] = row[f"{stim_src_id}_post_used_modif"]
				new_row[exercise_disrupt_concentration_col] = row[f"{stim_src_id}_post_exercise_1"]
				new_row[exercise_help_concentration_col] = row[f"{stim_src_id}_post_exercise_2"]
				new_row[exercise_more_prepared_col] = row[f"{stim_src_id}_post_exercise_3"]
				new_row[exercise_less_prepared_col] = row[f"{stim_src_id}_post_exercise_4"]

			if new_row[ai_col]:
				new_row[read_ai_col] = row[f"{stim_src_id}_post_used_ai"]
				new_row[ai_deceptive_col] = row[f"{stim_src_id}_post_ai_1"]
				new_row[ai_suspicious_col] = row[f"{stim_src_id}_post_ai_2"]
				new_row[ai_confident_col] = row[f"{stim_src_id}_post_ai_3"]
				new_row[ai_dependable_col] = row[f"{stim_src_id}_post_ai_4"]

			new_row[satisfied_ease_col] =  row[f"{stim_src_id}_post_task_1"]
			new_row[satisfied_time_col] =  row[f"{stim_src_id}_post_task_2"]
			new_row[confidence_col] =  row[f"{stim_src_id}_post_confidence_1"]
			# Convert to numerical value
			if new_row[confidence_col] == "Not at all confident":
				new_row[confidence_col] = "1"
			elif new_row[confidence_col] == "Extremely confident":
				new_row[confidence_col] = "7"

			if (isinstance(new_row[confidence_col], str)):
				new_row[confidence_col] = int(new_row[confidence_col])

			likert_dict = {"Strongly disagree": 1, "Somewhat disagree": 2, "Neither agree nor disagree": 3, "Somewhat agree": 4, "Strongly agree": 5, np.nan: np.nan}
			if new_row[exercise_col]:
				new_row[exercise_disrupt_concentration_col] = likert_dict[new_row[exercise_disrupt_concentration_col]]
				new_row[exercise_help_concentration_col] = likert_dict[new_row[exercise_help_concentration_col]]
				new_row[exercise_more_prepared_col] = likert_dict[new_row[exercise_more_prepared_col]]
				new_row[exercise_less_prepared_col] = likert_dict[new_row[exercise_less_prepared_col]]

			if new_row[ai_col]:
				new_row[ai_deceptive_col] = likert_dict[new_row[ai_deceptive_col]]
				new_row[ai_suspicious_col] = likert_dict[new_row[ai_suspicious_col]]
				new_row[ai_confident_col] = likert_dict[new_row[ai_confident_col]]
				new_row[ai_dependable_col] = likert_dict[new_row[ai_dependable_col]]



			# Demographic questions
			# TODO race/ethnic _8_TEXT, gender _5_TEXT, occupation _7_TEXT, describe_prog_contex_9_TEXT, 
			for s in [race_ethnic_col, gender_col, occupation_col, occupation_other_col, um_year_col, prog_exp_1_to_10_col, describe_prog_context_col, approx_writing_code_col, level_of_education_col]:
				new_row[s] = row[s]

			for s in [days_vigorous_col, minutes_vigorous_col, days_moderate_col, minutes_moderate_col, days_walking_col, minutes_walking_col, hours_sitting_col]:
				new_row[s] = row[s]

			for s in [language_at_home_col, language_prefer_info_col, how_well_speak_engl_col, how_well_read_engl_col]:
				new_row[s] = row[s]
			
			for i in range(1,6):
				new_row["list_languages_" + str(i)] = row["list_languages_" + str(i)]
				new_row["list_lang_percents_" + str(i)] = row["list_lang_percents_" + str(i)]

			new_rows.append(new_row)

			# Special cases
			# if (new_row[part_id_col] == "FJL2") and (i == 0):
			# 	new_row[exercise_col] = False
			# if (new_row[part_id_col] == "AMF1") and (i == 3 or i == 5):
			# 	new_row[exercise_col] = False

	transformed_df = pd.DataFrame(new_rows)

	return transformed_df

def extract_defect_type(path):
	# stimuli/defect-seeded/fast-excel/AbstractExcelWriteExecutor_fillComment_ExtraAssignment.java
	parts = path.split("/")
	if parts[1] == "clean":
		return np.nan
	
	filename = parts[len(parts)-1]
	filename = filename.split(".")[0] # get rid of .java
	return filename.split("_")[2]

def get_method_name(path):
	# stimuli/defect-seeded/fast-excel/AbstractExcelWriteExecutor_fillComment_ExtraAssignment.java
    # Remove directory
    filename = path.split("/")[-1]
    # Remove extension
    filename = filename.replace(".java", "")
    # Extract part after last underscore (e.g., buildChain)
    if "_" in filename:
        return filename.split("_")[1]
    return filename

def get_project_name(path):
	return path.split("/")[2]

def add_stimuli_info(df):
	defect_type_df = pd.read_csv("stimuli/stimulus_to_num.csv")
	stim_len_df = pd.read_csv("data_management/stimuli_misc_info/stimuli_info.csv")
	cyclomatic_df = pd.read_csv("data_management/stimuli_misc_info/lizard_cyclomatic.csv")
	halstead_df = pd.read_csv("data_management/stimuli_misc_info/halstead_complexity.csv")

	# -------------- Add defect type --------------
	# stimuli/clean/spring-ai-alibaba/ObservableToolCallingManager_executeToolCall.java, 15
	# stimuli/defect-seeded/fast-excel/AbstractExcelWriteExecutor_fillComment_ExtraAssignment.java, 14
	defect_type_df[defect_type_col] = defect_type_df[file_path_col].apply(extract_defect_type)

	print(defect_type_df)
	defect_type_df[stim_src_id_col] = defect_type_df[stim_src_id_col].astype(str) # convert to str to match in merge()

	# Merge defect type column into df based on stim src id
	df = df.merge(
		defect_type_df[[stim_src_id_col, file_path_col, defect_type_col]],
		on=stim_src_id_col,
		how="left"
	)

	# # -------------- Add stimulus length --------------
	# # stimuli/clean/fast-excel stimuli/clean/fast-excel/AbstractExcelWriteExecutor_fillComment.java,47,343
	# # stimuli/clean/fast-excel/AbstractWriteHolder_buildChain.java,54,527
	# stim_len_df[method_name_col] = stim_len_df[file_path_col].apply(get_method_name)

	# # Merge stimulus length cols into df based on filename
	# df = df.merge(
	# 	stim_len_df[[method_loc_col, file_loc_col, method_name_col]],
	# 	on=method_name_col,
	# 	how="left"
	# )

	# -------------- Add per-method cyclomatic complexity --------------
	# stimuli/clean/spring-ai-alibaba/ObservableToolCallingManager_executeToolCall.java, 15
	# stimuli/defect-seeded/fast-excel/AbstractExcelWriteExecutor_fillComment_ExtraAssignment.java, 14
	df[method_name_col] = df[file_path_col].apply(get_method_name)

	cyclomatic_df[method_name_col] = cyclomatic_df[file_path_col].apply(get_method_name)

	# Merge cyclomatic complexity column into df based on method name
	df = df.merge(
		cyclomatic_df[[method_loc_col, file_loc_col, cyclomatic_compl_col, method_name_col]],
		on=method_name_col,
		how="left"
	)
	# -------------- Add per-method halstead complexity --------------
	halstead_df[method_name_col] = halstead_df[file_path_col].apply(get_method_name)

	# Merge halstead metrics into df based on method name
	df = df.merge(
		halstead_df[[halstead_vocabulary_col,halstead_length_col,halstead_volume_col,halstead_difficulty_col,halstead_effort_col,halstead_time_seconds_col,halstead_estimated_bugs_col,method_name_col]],
		on=method_name_col,
		how="left"
	)

	# -------------- Add project (spring-ai-alibaba or fast-excel) --------------
	df[project_name_col] = df[file_path_col].apply(get_project_name)

	return df

def get_demographics(df):
	num_woman = 0
	num_man = 0
	num_gender_prefer_not_say = 0

	num_asian = 0
	num_caucasian = 0
	num_black = 0
	num_latino = 0
	num_multiracial = 0
	num_race_prefer_not_say = 0

	num_undergrad_cs = 0
	num_undergrad_not_cs = 0
	num_grad_cs = 0
	num_grad_not_cs = 0
	num_job_cs = 0
	num_occupation_other = 0

	num_year_1 = 0
	num_year_2 = 0
	num_year_3 = 0
	num_year_4 = 0
	num_year_5_plus = 0

	num_12_months = 0
	num_1_to_2 = 0
	num_2_to_3 = 0
	num_3_to_4 = 0
	num_4_to_5 = 0
	num_5_to_6 = 0
	num_6_to_7 = 0
	num_7_to_8 = 0
	num_9_over = 0

	dict_zoom_self = {"zoom": 0, "self": 0}

	dict_self_reported_exp = {}

	# for s in [race_ethnic_col, gender_col, occupation_col, um_year_col, prog_exp_1_to_10_col, describe_prog_context_col, approx_writing_code_col, level_of_education_col]:

	for idx, row in df.iterrows():
		part_id = row["part_id"]

		if (not isinstance(part_id, str)):
			continue
		if (not re.match(r"^[a-zA-Z]{3}\d$", part_id)):
			continue
		if (part_id == "PRS3"):
			continue

		match row[gender_col]:
			case "Woman":
				num_woman += 1
			case "Man":
				num_man += 1
			case "Prefer not to say":
				num_gender_prefer_not_say += 1

		if "Asian" in row[race_ethnic_col]:
			num_asian += 1
		if "Caucasian" in row[race_ethnic_col]:
			num_caucasian += 1
		if "Black" in row[race_ethnic_col]:
			num_black += 1
		if "Latino" in row[race_ethnic_col]:
			num_latino += 1
		if "Multiracial" in row[race_ethnic_col]:
			num_multiracial += 1
		if "Prefer" in row[race_ethnic_col]:
			num_race_prefer_not_say += 1

		if "Undergraduate in" in row[occupation_col]:
			num_undergrad_cs += 1
		elif "Undergraduate not" in row[occupation_col]:
			num_undergrad_not_cs += 1
		elif "Graduate student in" in row[occupation_col]:
			num_grad_cs += 1
		elif "Graduate student not" in row[occupation_col]:
			num_grad_not_cs += 1
		elif "job" in row[occupation_col]:
			num_job_cs += 1
		elif "Other" in row[occupation_col]:
			num_occupation_other += 1

		if isinstance(row[um_year_col], str):
			if "1" in row[um_year_col]:
				num_year_1 += 1
			elif "2" in row[um_year_col]:
				num_year_2 += 1
			elif "3" in row[um_year_col]:
				num_year_3 += 1
			elif "4" in row[um_year_col]:
				num_year_4 += 1
			elif "5" in row[um_year_col]:
				num_year_5_plus += 1

		if "months" in row[approx_writing_code_col]:
			num_12_months += 1
		# elif "1-" in row[approx_writing_code_col]:
		elif "2 years" in row[approx_writing_code_col]:
			num_1_to_2 += 1
		# elif "2-" in row[approx_writing_code_col]:
		elif "3 years" in row[approx_writing_code_col]:
			num_2_to_3 += 1
		elif "3-" in row[approx_writing_code_col]:
			num_3_to_4 += 1
		elif "4-" in row[approx_writing_code_col]:
			num_4_to_5 += 1
		elif "5" in row[approx_writing_code_col]:
			num_5_to_6 += 1
		elif "6-" in row[approx_writing_code_col]:
			num_6_to_7 += 1
		elif "7-" in row[approx_writing_code_col]:
			num_7_to_8 += 1
		elif "Over" in row[approx_writing_code_col]:
			num_9_over += 1
		else:
			print("SOMETHING ELSE *** ", row[approx_writing_code_col])

		if "Zoom" in row["selected_format"]:
			dict_zoom_self["zoom"] += 1
		elif "Self" in row["selected_format"]:
			dict_zoom_self["self"] += 1

		dict_self_reported_exp[row[prog_exp_1_to_10_col]] = dict_self_reported_exp.get(row[prog_exp_1_to_10_col], 0) + 1

	print(f"num woman {num_woman}, num man {num_man}, num prefer not to say {num_gender_prefer_not_say}")
	print(f"num asian {num_asian}, num caucasian {num_caucasian}, num black {num_black}, num latino {num_latino}, num multiracial {num_multiracial}, num prefer not to say {num_race_prefer_not_say}")
	print(f"num undergrad cs {num_undergrad_cs}, num undergrad not cs {num_undergrad_not_cs}, num grad cs {num_grad_cs}, num grad not cs {num_grad_not_cs}, num job cs {num_job_cs}, num other {num_occupation_other}")
	print(f"num year 1 {num_year_1}, num year 2 {num_year_2}, num year 3 {num_year_3}, num year 4 {num_year_4}, num year 5+ {num_year_5_plus}")
	print(f"num 3-12 month {num_12_months}, num 1-2 {num_1_to_2}, num 2-3 {num_2_to_3}, num 3-4 {num_3_to_4}, num 4-5 {num_4_to_5}, num 5-6 {num_5_to_6}, num 6-7 {num_6_to_7} num 7-8 {num_7_to_8} num 9 over {num_9_over}")
	print("dict_zoom_self:",dict_zoom_self)	
	print("dict_self_reported_exp:",dict_self_reported_exp)



def filter_dataframe(df):
	df = df[df[summary_q_len_col] > 0] # Keep rows where summary length is > 0
	print("removed rows with summary len 0:")
	print(len(df)) 

	# # Compute mean and SD of whole column
	# duration_sd = df[duration_col].std()
	# duration_mean = df[duration_col].mean()
	# df = df[ (df[duration_col] >= (duration_mean - 2*duration_sd)) & 
	# 	 	 (df[duration_col] <= (duration_mean + 2*duration_sd))]

	# Compute mean and SD of column based on experimental conditions
	# Keep rows that are within 2 SDs of mean for that experimental condition
	duration_means = df.groupby(condition1_col)[duration_col].transform('mean')
	duration_stds = df.groupby(condition1_col)[duration_col].transform('std')
	print("duration_means:")
	print(duration_means)
	print("len:",len(duration_means))
	print("duration_stds:")
	print(duration_stds)
	# for x in duration_means:
	# 	print(x)

	mask = (df[duration_col] >= (duration_means - 2*duration_stds)) & (df[duration_col] <= (duration_means + 2*duration_stds))

	print("Rows kept:", mask.sum())
	print("Rows removed:", len(mask) - mask.sum())


	df = df[ (df[duration_col] >= (duration_means - 2*duration_stds)) & 
		 	 (df[duration_col] <= (duration_means + 2*duration_stds))]
	
	# Keep rows where participants did not skip exercise break
	df = df[ ~( 
		(df[part_id_col] == "ESA6") & (df[stim_src_id_col] == '30')
	)]
	df = df[~(
		(df[part_id_col] == "FJL2") & (df[stim_src_id_col] == '14')
	)]
	df = df[~(
		(df[part_id_col] == "AMF1") & (df[stim_src_id_col] == '11')
	)]
	df = df[~(
		(df[part_id_col] == "AMF1") & (df[stim_src_id_col] == '9')
	)]
	# Keep rows where participants followed directions (i.e. summarized correct method)
	df = df[~(
		(df[part_id_col] == "ESQ3") & (df[stim_src_id_col] == '10')
	)]
	df = df[~(
		(df[part_id_col] == "ESQ3") & (df[stim_src_id_col] == '17')
	)]
	df = df[~(
		(df[part_id_col] == "ESQ3") & (df[stim_src_id_col] == '20')
	)]
	df = df[~(
		(df[part_id_col] == "ESQ3") & (df[stim_src_id_col] == '25')
	)]
	df = df[~(
		(df[part_id_col] == "ESQ3") & (df[stim_src_id_col] == '28')
	)]

	# for idx, row in df.iterrows():
	# 	if row[summary_q_len_col] == 0:
	# 		df = df.drop()

	return df


def check_transformed_dataframe(df):
	print("number of rows =",len(df))
	print("n =",len(df) / 6)
	print("Average duration of 1 task:", df[duration_col].mean())

	# print(df.head())

	num_exercise_ai_1 = 0
	num_exercise_no_ai_1 = 0
	num_no_exercise_ai_1 = 0
	num_no_exercise_no_ai_1 = 0
	num_exercise_ai_2 = 0
	num_exercise_no_ai_2 = 0
	num_no_exercise_ai_2 = 0
	num_no_exercise_no_ai_2 = 0
	num_tp = 0 
	num_tn = 0
	num_fp = 0
	num_fn = 0
	num_ai = 0
	num_no_ai = 0
	num_exercise = 0
	num_no_exercise = 0

	part_exercise_ai = dict()
	part_exercise_no_ai = dict()
	part_no_exercise_ai = dict()
	part_no_exercise_no_ai = dict()
	
	for idx, row in df.iterrows():
		match row[tp_tn_col]:
			case "TP":
				num_tp += 1
			case "TN":
				num_tn += 1
			case "FP":
				num_fp += 1
			case "FN":
				num_fn += 1

		match row[condition1_col]:
			case "exercise_ai":
				num_exercise_ai_1 += 1
				part_exercise_ai[row[part_id_col]] = True
			case "exercise_no_ai":
				num_exercise_no_ai_1 += 1
				part_exercise_no_ai[row[part_id_col]] = True
			case "no_exercise_ai":
				num_no_exercise_ai_1 += 1
				part_no_exercise_ai[row[part_id_col]] = True
			case "no_exercise_no_ai":
				num_no_exercise_no_ai_1 += 1
				part_no_exercise_no_ai[row[part_id_col]] = True
		match row[condition2_col]:
			case "exercise_ai":
				num_exercise_ai_2 += 1
			case "exercise_no_ai":
				num_exercise_no_ai_2 += 1
			case "no_exercise_ai":
				num_no_exercise_ai_2 += 1
			case "no_exercise_no_ai":
				num_no_exercise_no_ai_2 += 1

		if row[ai_col]:
			num_ai += 1
		else:
			num_no_ai += 1
		
		if row[exercise_col]:
			num_exercise += 1
		else:
			num_no_exercise += 1

	print(f" num_tp {num_tp}\n num_tn {num_tn}\n num_fp {num_fp}\n num_fn {num_fn}\n num_ai {num_ai}\n num_no_ai {num_no_ai}\n num_exercise {num_exercise}\n num_no_exercise {num_no_exercise}")
	print()

	print(f" num_exercise_ai_1 {num_exercise_ai_1}   {len(part_exercise_ai)}\n num_exercise_no_ai_1 {num_exercise_no_ai_1}  {len(part_exercise_no_ai)}\n num_no_exercise_ai_1 {num_no_exercise_ai_1}   {len(part_no_exercise_ai)}\n num_no_exercise_no_ai_1 {num_no_exercise_no_ai_1}  {len(part_no_exercise_no_ai)}")
	print(f" num_exercise_ai_2 {num_exercise_ai_2}\n num_exercise_no_ai_2 {num_exercise_no_ai_2}\n num_no_exercise_ai_2 {num_no_exercise_ai_2}\n num_no_exercise_no_ai_2 {num_no_exercise_no_ai_2}")


	print()

	part_num_buggy = dict()
	part_num_tasks = dict()
	print(part_id_col, stim_src_id_col, has_bug_col, stim_order_col, condition1_col, exercise_col, ai_col, tp_tn_col, duration_col, summary_q_len_col)
	for idx, row in df.iterrows():
		# print(row[part_id_col], row[stim_src_id_col], row[has_bug_col], row[stim_order_col], row[condition1_col], row[exercise_col], row[ai_col], row[tp_tn_col], row[duration_col], row[summary_q_len_col])
		part_num_tasks[row[part_id_col]] = part_num_tasks.get(row[part_id_col], 0) + 1
		if row[has_bug_col]:
			part_num_buggy[row[part_id_col]] = part_num_buggy.get(row[part_id_col], 0) + 1
	print("part_num_tasks:",part_num_tasks)
	print("part_num_buggy:",part_num_buggy)
	# for key in part_num_buggy:
	# 	# assert(part_num_buggy[key] == 3)
	# print(len(part_num_buggy))

	print()

	stimuli_counts = dict()

	for idx, row in df.iterrows():
		stimuli_counts[row[stim_src_id_col]] = stimuli_counts.get(row[stim_src_id_col], 0) + 1
	print("Stimuli counts:")
	print(stimuli_counts)
	print(stimuli_counts.values())
	print(sum(stimuli_counts.values()))
	print()


def visualize_data(df):
	# # Create a boxplot for duration
	# plt.boxplot(df[duration_col])
	# # plt.boxplot(df[duration_col], labels=['Group A', 'Group B', 'Group C'])

	# plt.title('Boxplot of Multiple Groups')
	# plt.xlabel('Group')
	# plt.ylabel('Value')
	# plt.show()

	# Subsets
	ai_true = df[df[ai_col] == True][duration_col]
	ai_false = df[df[ai_col] == False][duration_col]
	ex_true = df[df[exercise_col] == True][duration_col]
	ex_false = df[df[exercise_col] == False][duration_col]

	exercise_ai_dur = df[df[exercise_col] & df[ai_col]][duration_col]
	exercise_no_ai_dur = df[df[exercise_col] & ~df[ai_col]][duration_col]
	no_exercise_ai_dur = df[~df[exercise_col] & df[ai_col]][duration_col]
	no_exercise_no_ai_dur = df[~df[exercise_col] & ~df[ai_col]][duration_col]

	print()
	print("Means and SD:")
	print("Exercise and AI: ", np.mean(exercise_ai_dur), np.std(exercise_ai_dur))
	print("Exercise and no AI: ", np.mean(exercise_no_ai_dur), np.std(exercise_no_ai_dur))
	print("No exercise and AI: ", np.mean(no_exercise_ai_dur), np.std(no_exercise_ai_dur))
	print("No exercise and no AI: ", np.mean(no_exercise_no_ai_dur), np.std(no_exercise_no_ai_dur))
	print()

	# Box plot
	plt.figure(figsize=(10, 6))

	plt.boxplot(
		[exercise_ai_dur, exercise_no_ai_dur, no_exercise_ai_dur, no_exercise_no_ai_dur],
		labels=["Exercise and AI", "Exercise and no AI", "No exercise and AI", "No exercise and no AI"], 
		showfliers=False
	)


	for i, g in enumerate([ai_true, ai_false, ex_true, ex_false], start=1):
		# Jitter around x-position i
		jitter = np.random.normal(0, 0.02, size=len(g))
		plt.scatter(np.full(len(g), i) + jitter, g, s=12, alpha=0.5)


	plt.ylabel("Response time (sec)")
	# plt.title("Duration Boxplots by AI and Exercise Conditions")
	plt.show()

	# Histogram with different colors for experimental conditions
	plt.figure(figsize=(10, 6))

	plt.hist(exercise_ai_dur, bins=20, alpha=0.4, label="Exercise and AI")
	plt.hist(exercise_no_ai_dur, bins=20, alpha=0.4, label="Exercise and no AI")
	plt.hist(no_exercise_ai_dur, bins=20, alpha=0.4, label="No exercise and AI")
	plt.hist(no_exercise_no_ai_dur, bins=20, alpha=0.4, label="No exercise and no AI")

	plt.xlabel("Duration")
	plt.ylabel("Count")
	plt.title("Histogram of Duration for All Conditions")
	plt.legend()
	plt.show()

	# 2x2 plot of histograms
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	for ax, (title, g) in zip(axes.flat, [["Exercise and AI", exercise_ai_dur], ["Exercise and no AI",exercise_no_ai_dur], ["No exercise and AI",no_exercise_ai_dur], ["No exercise and no AI",no_exercise_no_ai_dur]]):
		ax.hist(g, bins=20, alpha=0.7)
		ax.set_title(title)
		ax.set_xlabel("Duration")
		ax.set_ylabel("Count")

	fig.suptitle("Histograms of Duration by Condition")
	plt.tight_layout()
	plt.show()

	# Violin plot
	plt.figure(figsize=(10, 6))

	plt.violinplot(
		[ai_true, ai_false, ex_true, ex_false],
		showmeans=True,
		showextrema=True,
	)

	for i, g in enumerate([ai_true, ai_false, ex_true, ex_false], start=1):
		# Jitter around x-position i
		jitter = np.random.normal(0, 0.02, size=len(g))
		plt.scatter(np.full(len(g), i) + jitter, g, s=12, alpha=0.5)

	plt.xticks(
		[1, 2, 3, 4],
		["AI condition", "No AI condition", "Exercise condition", "No exercise condition"]
	)

	plt.ylabel("Response time (sec)")
	# plt.title("Violin Plots of Duration by AI & Exercise Conditions")
	plt.show()


	if False:
		df[duration_col].plot(kind='hist', bins=50)
		plt.show()

def t_tests_transformed_dataframe(df):
	# print("good ai")
	# for idx, row in df.iterrows():
	# 	if idx <= 43*6:
	# 		if row[ai_col] and (row[tp_tn_col] == TP or row[tp_tn_col] == TN):
	# 			print(row[duration_col])

	# print()
	# print("bad ai")
	# for idx, row in df.iterrows():
	# 	if idx <= 43*6:
	# 		if row[ai_col] and (row[tp_tn_col] == FP or row[tp_tn_col] == FN):
	# 			print(row[duration_col])
		

	print("df length:",len(df))
	print("ai")
	for idx, row in df.iterrows():
		if idx <= 47*6 and row[zoom_self_col] == "zoom":
			if row[ai_col]:
				print(row[duration_col])

	print()
	print("not ai")
	for idx, row in df.iterrows():
		if idx <= 47*6 and row[zoom_self_col] == "zoom":
			if not row[ai_col]:
				print(row[duration_col])

def merge_qualitative(df, qualitative_csv_1, qualitative_csv_2):
	# Read in CSV files as dataframes
	scores_df1 = pd.read_csv(qualitative_csv_1)
	scores_df2 = pd.read_csv(qualitative_csv_2)

	# List of quality attributes 
	qual_categories = [readability_col, completeness_col, conciseness_col,accuracy_col]
	
	# -------------- Add qualitative analysis annotations --------------
	scores_df1[stim_src_id_col] = scores_df1[stim_src_id_col].astype(str) # convert stim_id to string so we can match during merge()
	scores_df2[stim_src_id_col] = scores_df2[stim_src_id_col].astype(str) 

	scores_df1 = scores_df1.rename(columns={col: f"{col}_scorer_1" for col in qual_categories})
	scores_df1 = scores_df1.rename(columns={reasoning_col: reasoning_col+"_scorer_1"})
	scores_df1 = scores_df1.rename(columns={themes_col: themes_col+"_scorer_1"})
	scores_df2 = scores_df2.rename(columns={col: f"{col}_scorer_2" for col in qual_categories})
	scores_df2 = scores_df2.rename(columns={reasoning_col: reasoning_col+"_scorer_2"})
	scores_df2 = scores_df2.rename(columns={themes_col: themes_col+"_scorer_2"})

	# Merge 4 quality categories from scorer dataframes into df
	# df = df.merge(
	# 	scores_df1[[part_id_col, stim_src_id_col] + [f"{col}_scorer_1" for col in qual_categories]],
	# 	on=[part_id_col, stim_src_id_col],
	# 	how="left"
	# )


	df = df.merge(
		scores_df1[[part_id_col, stim_src_id_col] + [f"{col}_scorer_1" for col in qual_categories] + [f"{reasoning_col}_scorer_1", f"{themes_col}_scorer_1"]],
		on=[part_id_col, stim_src_id_col],
		how="left"
	)
	df = df.merge(
		scores_df2[[part_id_col, stim_src_id_col] + [f"{col}_scorer_2" for col in qual_categories] + [f"{reasoning_col}_scorer_2", f"{themes_col}_scorer_2"]],
		on=[part_id_col, stim_src_id_col],
		how="left"
	)

	# -------------- Add column for average of scorer 1 and scorer 2 --------------
	for cat in qual_categories:
		print(df[f"{cat}_scorer_1"] )
		print(df[f"{cat}_scorer_2"] )
		df[f"{cat}_both"] = ( df[f"{cat}_scorer_1"] + df[f"{cat}_scorer_2"] ) / 2

	# -------------- Use bug description scoring column from scorer 2 --------------
	scores_df2 = scores_df2.rename(columns={correctly_identifies_bug_col: correctly_identifies_bug_col+"_scorer_2"})

	df = df.merge(
		scores_df2[[part_id_col, stim_src_id_col, correctly_identifies_bug_col+"_scorer_2"]],
		on=[part_id_col, stim_src_id_col],
		how="left"
	)

	# -------------- Add column for average of scorer 1 and scorer 2 --------------


	print("merged df.head()")
	print(df.head())

	return df
	

def analyze(df):
	# Averages
	# Duration
	print("Average duration of 1 task:", df[duration_col].mean())
	print("Average duration of 1 task if exercise:", df.loc[df[exercise_col] == True, duration_col].mean())
	print("SD", np.std(df.loc[df[exercise_col] == True, duration_col]))
	print("Average duration of 1 task if no exercise:", df.loc[df[exercise_col] == False, duration_col].mean())
	print("SD", np.std(df.loc[df[exercise_col] == False, duration_col]))
	print("Average duration of 1 task if ai:", df.loc[df[ai_col] == True, duration_col].mean())
	print()

	print("Average duration of 1 task if good ai:", df.loc[ (df[ai_col]) & ( (df[tp_tn_col] == TP) | (df[tp_tn_col] == TN) ), duration_col].mean())
	print("Average duration of 1 task if bad ai:", df.loc[ (df[ai_col]) & ( (df[tp_tn_col] == FP) | (df[tp_tn_col] == FN) ), duration_col].mean())
	print()

	print("Average duration of 1 task if exercise and ai:", df.loc[ (df[exercise_col] & df[ai_col]), duration_col].mean())
	print("Average duration of 1 task if exercise and no ai:", df.loc[ (df[exercise_col] & ~df[ai_col]), duration_col].mean())
	print("Average duration of 1 task if no exercise and ai:", df.loc[ (~df[exercise_col] & df[ai_col]), duration_col].mean())
	print("Average duration of 1 task if no exercise and no ai:", df.loc[(~df[exercise_col]) & ~(df[ai_col]), duration_col].mean())
	print()

	# Length of summary
	print("Average length of 1 summary:", df[summary_q_len_col].mean())
	print("Average length of 1 summary if exercise:", df.loc[df[exercise_col], summary_q_len_col].mean())
	print("SD:",np.std(df.loc[df[exercise_col], summary_q_len_col]))
	print("Average length of 1 summary if no exercise:", df.loc[df[exercise_col]==False, summary_q_len_col].mean())
	print("SD:", np.std(df.loc[df[exercise_col]==False, summary_q_len_col]))
	print("Average length of 1 summary if ai:", df.loc[df[ai_col], summary_q_len_col].mean())
	print("SD:", np.std(df.loc[df[ai_col], summary_q_len_col]))
	print("Average length of 1 summary if no ai:", df.loc[df[ai_col]==False, summary_q_len_col].mean())
	print("SD:", np.std(df.loc[df[ai_col]==False, summary_q_len_col]))
	print()

	print("Average length of 1 summary if good ai:", df.loc[ (df[ai_col]) & ( (df[tp_tn_col] == TP) | (df[tp_tn_col] == TN) ), summary_q_len_col].mean())
	print("Average length of 1 summary if bad ai:", df.loc[ (df[ai_col]) & ( (df[tp_tn_col] == FP) | (df[tp_tn_col] == FN) ), summary_q_len_col].mean())
	print()

	print("Average length of 1 summary if exercise and ai:", df.loc[ (df[exercise_col] & df[ai_col]), summary_q_len_col].mean())
	print("Average length of 1 summary if exercise and no ai:", df.loc[ (df[exercise_col] & ~df[ai_col]), summary_q_len_col].mean())
	print("Average length of 1 summary if no exercise and ai:", df.loc[ (~df[exercise_col] & df[ai_col]), summary_q_len_col].mean())
	print("Average length of 1 summary if no exercise and no ai:", df.loc[(~df[exercise_col]) & ~(df[ai_col]), summary_q_len_col].mean())
	print()

	# T-tests -- self-confidence
	t_statistic, p_value = stats.ttest_ind(
		df.loc[df[exercise_col], confidence_col].dropna(), 
		df.loc[~df[exercise_col], confidence_col].dropna(), 
		equal_var=False, 
		alternative='greater'
	)
	print("T test, 1-tailed: self-confidence, exercise vs no exercise. t_statistic:",t_statistic, "p_value:",p_value)

	print("exercise disrupts concentration, exercise. average:",df.loc[df[exercise_col], exercise_disrupt_concentration_col].mean())
	print("exercise helps concentration, exercise. average:",df.loc[df[exercise_col], exercise_help_concentration_col].mean())
	print("exercise less prepared, exercise. average:",df.loc[df[exercise_col], exercise_less_prepared_col].mean())
	print("exercise more prepared, exercise. average:",df.loc[df[exercise_col], exercise_more_prepared_col].mean())

	for col in [exercise_disrupt_concentration_col, exercise_help_concentration_col, exercise_less_prepared_col, exercise_more_prepared_col]:
		counts_exercise = {}
		for val in df.loc[df[exercise_col], col]:
			counts_exercise[val] = counts_exercise.get(val, 0) + 1
		print(col, "exercise. counts:",counts_exercise)


	for col in [ai_deceptive_col, ai_dependable_col, ai_confident_col, ai_suspicious_col]:
		counts_ai = {}
		for val in df.loc[df[ai_col], col]:
			counts_ai[val] = counts_ai.get(val, 0) + 1
		print(col, "counts:",counts_ai)
	
	t_statistic, p_value = stats.ttest_ind(
		df.loc[df[ai_col], confidence_col].dropna(), 
		df.loc[~df[ai_col], confidence_col].dropna(), 
		equal_var=False, 
		alternative='greater'
	)
	print("T test, 1-tailed: self-confidence, ai vs no ai. t_statistic:",t_statistic, "p_value:",p_value)
	
	# T-tests -- duration
	t_statistic, p_value = stats.ttest_ind(
		df.loc[df[ai_col], duration_col], 
		df.loc[~df[ai_col], duration_col], 
		equal_var=False, 
		alternative='less'
	)
	print("T test, 1-tailed: duration, ai vs no ai. t_statistic:",t_statistic, "p_value:",p_value)
	
	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[ai_col]) & (df[zoom_self_col] == "zoom"), duration_col], 
		df.loc[(~df[ai_col]) & (df[zoom_self_col] == "zoom"), duration_col], 
		equal_var=False,
		alternative='less'
	)
	print("T test, 1-tailed: duration, ai vs no ai AND zoom-supervised. t_statistic:",t_statistic, "p_value:",p_value)
	
	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[ai_col]) & ((df[tp_tn_col] == TP) | (df[tp_tn_col] == TN)), duration_col], 
		df.loc[(df[ai_col]) & ((df[tp_tn_col] == FP) | (df[tp_tn_col] == FN)), duration_col],
		equal_var=False
	)
	print("T test: duration, good ai vs bad ai. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]), duration_col], 
		df.loc[(~df[exercise_col]), duration_col],
		equal_var=False
	)
	print("T test: duration, exercise vs no exercise. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]) & (df[zoom_self_col] == "zoom"), duration_col], 
		df.loc[(~df[exercise_col]) & (df[zoom_self_col] == "zoom"), duration_col],
		equal_var=False
	)
	print("T test: duration, exercise vs no exercise AND zoom-supervised. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]), duration_col], 
		df.loc[(~df[exercise_col]), duration_col],
		equal_var=False,
		alternative="greater"
	)
	print("T test, 1-tailed: duration, exercise vs no exercise. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]) & (df[zoom_self_col] == "zoom"), duration_col], 
		df.loc[(~df[exercise_col]) & (df[zoom_self_col] == "zoom"), duration_col],
		equal_var=False,
		alternative="greater"
	)
	print("T test, 1-tailed: duration, exercise vs no exercise AND zoom-supervised. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col] & df[ai_col]), duration_col], 
		df.loc[(~df[exercise_col] & ~df[ai_col]), duration_col],
		equal_var=False
	)
	print("T test: duration, exercise and ai vs no exercise no ai. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[ (df[exercise_col]) & (df[ai_col]) & ( (df[tp_tn_col] == FP) | (df[tp_tn_col] == FN) ), duration_col],
		df.loc[(df[exercise_col]), duration_col], 
		equal_var=False
	)
	print("T test: duration, exercise and bad ai vs exercise. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[ (df[exercise_col]) & (df[ai_col]) & ( (df[tp_tn_col] == FP) | (df[tp_tn_col] == FN) ), duration_col],
		df.loc[(df[ai_col]) & ( (df[tp_tn_col] == FP) | (df[tp_tn_col] == FN) ), duration_col], 
		equal_var=False
	)
	print("T test: duration, exercise and bad ai vs bad ai. t_statistic:",t_statistic, "p_value:",p_value)
	print()

	# T-tests -- summary length
	t_statistic, p_value = stats.ttest_ind(
		df.loc[df[ai_col], summary_q_len_col], 
		df.loc[~df[ai_col], summary_q_len_col], 
		equal_var=False, 
		alternative='less'
	)
	print("T test, 1-tailed: summary len, ai vs no ai. t_statistic:",t_statistic, "p_value:",p_value)
	
	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[ai_col]) & (df[zoom_self_col] == "zoom"), summary_q_len_col], 
		df.loc[(~df[ai_col]) & (df[zoom_self_col] == "zoom"), summary_q_len_col], 
		equal_var=False,
		alternative='less'
	)
	print("T test, 1-tailed: summary len, ai vs no ai AND zoom-supervised. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]), summary_q_len_col], 
		df.loc[(~df[exercise_col]), summary_q_len_col],
		equal_var=False,
		alternative="greater"
	)
	print("T test, 1-tailed: summary len, exercise vs no exercise. t_statistic:",t_statistic, "p_value:",p_value)

	t_statistic, p_value = stats.ttest_ind(
		df.loc[(df[exercise_col]) & (df[zoom_self_col] == "zoom"), summary_q_len_col], 
		df.loc[(~df[exercise_col]) & (df[zoom_self_col] == "zoom"), summary_q_len_col],
		equal_var=False,
		alternative="greater"
	)
	print("T test, 1-tailed: summary len, exercise vs no exercise AND zoom-supervised. t_statistic:",t_statistic, "p_value:",p_value)
	print()


	# ANOVA
	f_statistic, p_value = stats.f_oneway( 
		df.loc[(df[exercise_col]) & (df[ai_col]), duration_col],
		df.loc[(df[exercise_col]) & (~df[ai_col]), duration_col],
		df.loc[(~df[exercise_col]) & (df[ai_col]), duration_col],
		df.loc[(~df[exercise_col]) & (~df[ai_col]), duration_col],
	)
	print("ANOVA: duration,", exercise_ai, exercise_no_ai, no_exercise_ai, no_exercise_no_ai, "f_statistic:", f_statistic, "p_value:",p_value)

	f_statistic, p_value = stats.f_oneway( 
		df.loc[(df[exercise_col]) & (df[ai_col]), summary_q_len_col],
		df.loc[(df[exercise_col]) & (~df[ai_col]), summary_q_len_col],
		df.loc[(~df[exercise_col]) & (df[ai_col]), summary_q_len_col],
		df.loc[(~df[exercise_col]) & (~df[ai_col]), summary_q_len_col],
	)
	print("ANOVA: summary length,", exercise_ai, exercise_no_ai, no_exercise_ai, no_exercise_no_ai, "f_statistic:", f_statistic, "p_value:",p_value)





def dataframe_to_csv(filepath: str, df):
	df.to_csv(filepath, index=False)


if __name__ == "__main__":
	# in_path = "analysis/data/Code Summarization + Exercise + ChatGPT_September 24, 2025_14.25.csv" # Preliminary analysis with 39 participants (and some excluded data within that)

	# Read in CSV file
	in_path = "analysis/data/Code Summarization + Exercise + ChatGPT_October 26, 2025_16.49.csv"
	original_df = pd.read_csv(in_path)
	
	# Transform dataframe
	df = transform_dataframe(original_df)

	df = add_stimuli_info(df)

	# Save unfiltered transformed CSV file
	unfiltered_out_path = "analysis/data/unfiltered_transformed_dataframe.csv"
	dataframe_to_csv(unfiltered_out_path, df)

	if debug:
		get_demographics(original_df)

	print("ORIGINAL LEN:",len(df))

	# Filter out participant responses
	df = filter_dataframe(df)
	print("FILTERED DATAFRAME LEN:",len(df))

	if debug:
		check_transformed_dataframe(df)

	if debug:
		visualize_data(df)

	if debug:
		t_tests_transformed_dataframe(df)

	# Run t-tests
	analyze(df)

	# Merge qualitative annotations with df
	scorer_csv_1 = "analysis/data/qualitative-scoring/Summaries from Participants - Rater-1 (no info about exp conditions).csv"
	scorer_csv_2 = "analysis/data/qualitative-scoring/Summaries from Participants - Rater-2 (no info about exp conditions).csv"
	df = merge_qualitative(df, scorer_csv_1, scorer_csv_2)

	# Save transformed dataframe as CSV file
	out_path = "analysis/data/transformed_dataframe.csv"
	dataframe_to_csv(out_path, df)