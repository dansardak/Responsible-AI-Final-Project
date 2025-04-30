# Analysis & Evaluation of Geographic Bias in Text-to-image generation

## Abstract 

Analysis & Evaluation of Geographic Bias in Text-to-image generation
Daniel Sardak, Krish Patel, Anurag Chinthalapati
Worcester Polytechnic Institute / Worcester, MA
{dsardak, kpatel, svenkata}@wpi.edu
Abstract
We introduce a comprehensive evaluation
framework to assess geographic bias in text-to-
image models across five global regions: North
America, South America, Europe, Africa, and
Asia. Using a combination of culturally diverse
prompts adapted from prior work, we gener-
ated image outputs from DALL·E 2, Stable Dif-
fusion, Gemini-2, and Kandinsky, each evalu-
ated ten times per prompt. Both no-reference
automatic image quality metrics (e.g., NIQE,
BRISQUE, CLIP-IQA) and human-annotated
scores (assessing regional authenticity, socioe-
conomic representation, and gender balance)
were used to quantify bias. Contrary to our hy-
pothesis, automatic metrics showed higher im-
age quality for African prompts, while human
evaluations revealed stereotypical and inaccu-
rate portrayals, particularly of Africa and parts
of Asia. These results suggest a disconnect be-
tween metric-based assessments and actual bias
perception. Limitations of the study include re-
liance on internal human raters and potentially
an incomplete picture of bias by focusing on
very large, general geographic areas


## File Overview

calcmetrics.py - For each of the files, calculate metrics
checkdallegenerated.py - Verify whether or not DALLE generation worked
excel_processor.ipynb - convert excel sheet of human metrics into one CSV
eval_auto_metrics.py - measure automatic metrics
generate_first_paper_prompts.py - generate prompts from DIG In paper
generate_second_paper_prompts.py - generate prompts from second paper
humanprocess.py - process human metrics
metricgui.py - create streamlit app to visualize metrics
metrictester.py - test file for metric implementation
runpaperone.py - generate images for paper 1
runpapertwo.py - generate images for paper 2
table_maker.ipynb - scratch file to generate visualizations
updated_eval_metrics.py - alternate metric evaluator
