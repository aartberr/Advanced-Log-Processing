import ollama
import re
import time
import string
import json
from .parsing_cache import ParsingCache
from .post_process import correct_single_template

""" Functions names were kept the same as gpt_query.py for easy replacement. You can replace the gpt_query.py with this file if you don't have access to openai API key."""

OLLAMA_CLIENT = ollama.Client()

def infer_llm(instruction, exemplars, query, log_message, model='llama3', temperature=0.0, max_tokens=2048):
    prompt_parts = [
        "You are an expert of log parsing, and now you will help to do log parsing.",
        instruction.strip(),
        ""
    ]

    if exemplars:
        prompt_parts.append("Here are some examples:")
        for exemplar in exemplars:
            prompt_parts.append(exemplar['query'])
            prompt_parts.append(exemplar['answer'])
        prompt_parts.append("")

    prompt_parts.append("Please parse the following log message:")
    prompt_parts.append(query)
    prompt = "\n".join(prompt_parts)

    retry_times = 0
    print("model:", model)

    while retry_times < 3:
        try:
            response = OLLAMA_CLIENT.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            text = response.get("response", "").strip()
            if text:
                return text
        except Exception as e:
            print("Exception:", e)
        retry_times += 1

    print(f"Failed to get response from Ollama after {retry_times} retries.")
    if exemplars is not None and len(exemplars) > 0:
        if exemplars[0]['query'] != 'Log message: `try to connected to host: 172.16.254.1, finished.`' \
        or exemplars[0]['answer'] != 'Log template: `try to connected to host: {ip_address}, finished.`':
            examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
            return infer_llm(instruction, examples, query, log_message, model, temperature, max_tokens)
    return 'Log message: `{}`'.format(log_message)


def get_response_from_openai_key(query, examples=None, model='llama3', temperature=0.0):
    # Prompt-1
    # instruction = "I want you to act like an expert of log parsing. I will give you a log message enclosed in backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template enclosed in backticks."
    instruction = "I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."
    if examples is None or len(examples) == 0:
        examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
    question = 'Log message: `{}`'.format(query)
    responses = infer_llm(instruction, examples, question, query,
                          model, temperature, max_tokens=2048)
    return responses


def query_template_from_gpt(log_message, examples=None, model='llama3'):
    if len(log_message.split()) == 1:
        return log_message, False
    # print("prompt base: ", prompt_base)
    response = get_response_from_openai_key(log_message, examples, model)
    # print(response)
    lines = response.split('\n')
    log_template = None
    for line in lines:
        if line.find("Log template:") != -1:
            log_template = line
            break
    if log_template is None:
        for line in lines:
            if line.find("`") != -1:
                log_template = line
                break
    if log_template is not None:
        start_index = log_template.find('`') + 1
        end_index = log_template.rfind('`')

        if start_index == 0 or end_index == -1:
            start_index = log_template.find('"') + 1
            end_index = log_template.rfind('"')

        if start_index != 0 and end_index != -1 and start_index < end_index:
            template = log_template[start_index:end_index]
            return template, True

    print("======================================")
    print("Ollama response format error: ")
    print(response)
    print("======================================")
    return log_message, False


def post_process_template(template, regs_common):
    pattern = r'\{(\w+)\}'
    template = re.sub(pattern, "<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(log_message, regs_common=[], examples=[], model="llama3"):
    template, flag = query_template_from_gpt(log_message, examples, model)
    if len(template) == 0 or flag == False:
        print(f"Ollama error")
    else:
        tree = ParsingCache()
        template, flag = post_process_template(template, regs_common)
        if flag:
            tree.add_templates(template)
            if tree.match_event(log_message)[0] == "NoMatch":
                print("==========================================================")
                print(log_message)
                print("Ollama template wrong: cannot match itself! And the wrong template is : ")
                print(template)
                print("==========================================================")
            else:
                return template, True
    return post_process_template(log_message, regs_common)
