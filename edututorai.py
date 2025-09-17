# -*- coding: utf-8 -*-
"""EduTutorAI

!pip install transformers torch gradio -q

# -*- coding: utf-8 -*-
"""EduTutorAI"""

!pip install transformers torch gradio -q

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Core generation function
def generate_response(prompt, max_new_tokens=600):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# Quiz generation with enforced answer format
def quiz_generator_with_answers(topic):
    prompt = (
        f"Generate exactly 5 multiple-choice quiz questions about {topic}. "
        f"Each question should have options labeled A), B), C), D). "
        f"At the end, include an ANSWERS section formatted like:\n"
        f"ANSWERS:\n1. A\n2. B\n3. C\n4. D\n5. A"
    )
    full_text = generate_response(prompt, max_new_tokens=600)

    if "ANSWERS" in full_text:
        questions_part, answers_part = full_text.split("ANSWERS", 1)
    else:
        questions_part, answers_part = full_text, ""

    questions = [q.strip() for q in questions_part.strip().split("\n") if q.strip()]
    answers = [a.strip() for a in answers_part.strip().split("\n") if a.strip()]
    return questions, answers

# Clean question blocks
def clean_questions(questions):
    cleaned = []
    skip = 0
    for i, line in enumerate(questions):
        if skip > 0:
            skip -= 1
            continue
        if line[0].isdigit() and line[1] == ".":
            block = line
            opt_lines = []
            for j in range(1, 5):
                if i + j < len(questions):
                    opt_line = questions[i + j].strip()
                    opt_lines.append(opt_line)
            block += "\n" + "\n".join(opt_lines)
            cleaned.append(block)
            skip = 4
    return cleaned

# Robust answer parser
def extract_option_letter(answer_line):
    if not answer_line:
        return "?"
    parts = answer_line.strip().split(".")
    if len(parts) == 2:
        letter = parts[1].strip().lower().replace(")", "")
        if letter in ["a", "b", "c", "d"]:
            return letter
    for option in ["A)", "B)", "C)", "D)"]:
        if option in answer_line:
            return option[0].lower()
    return "?"

# Answer checking logic
def check_answers(user_answers, correct_answers):
    score = 0
    feedback = []
    cleaned_correct = [extract_option_letter(ans) for ans in correct_answers]
    for i, (user_ans, correct_ans) in enumerate(zip(user_answers, cleaned_correct)):
        user_ans_clean = user_ans.strip().lower()
        if correct_ans == "?":
            feedback.append(f"Q{i+1}: ⚠️ No valid correct answer found.")
        elif user_ans_clean == correct_ans:
            feedback.append(f"Q{i+1}: ✅ Correct!")
            score += 1
        else:
            feedback.append(f"Q{i+1}: ❌ Incorrect! Correct answer is: {correct_ans.upper()}")
    feedback.append(f"\nFinal Score: {score}/{len(correct_answers)}")
    return "\n".join(feedback)

# Global state
correct_answers_global = []

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🧠 Educational AI Assistant")

    with gr.Tabs():
        with gr.TabItem("📘 Concept Explanation"):
            concept_input = gr.Textbox(label="Enter a concept", placeholder="e.g., machine learning")
            explain_btn = gr.Button("Explain")
            explanation_output = gr.Textbox(label="Explanation", lines=10)

            explain_btn.click(
                fn=lambda c: generate_response(
                    f"Explain the concept of {c} in detail with examples:", max_new_tokens=300
                ),
                inputs=concept_input,
                outputs=explanation_output,
            )

        with gr.TabItem("📝 Quiz Generator and Attender"):
            topic_input = gr.Textbox(label="Enter a topic", placeholder="e.g., Python programming")
            generate_btn = gr.Button("Generate Quiz")
            question_display = gr.Markdown()
            answer_boxes = [gr.Textbox(label=f"Answer Q{i+1}", lines=1) for i in range(5)]
            submit_btn = gr.Button("Submit Answers")
            result_display = gr.Textbox(label="Results", lines=10)

            def generate_quiz(topic):
                global correct_answers_global
                questions, answers = quiz_generator_with_answers(topic)
                questions = clean_questions(questions)
                correct_answers_global = answers
                question_markdown = ""
                for i, q in enumerate(questions):
                    question_markdown += f"**Q{i+1}.** {q}\n\n"
                return question_markdown

            def submit_answers(*user_answers):
                global correct_answers_global
                return check_answers(user_answers, correct_answers_global)

            generate_btn.click(fn=generate_quiz, inputs=topic_input, outputs=question_display)
            submit_btn.click(fn=submit_answers, inputs=answer_boxes, outputs=result_display)

app.launch(share=True)
