"""Paper prompts, benchmark records and answer scoring."""

import ast
import json
import operator
from pathlib import Path
import re
from datasets import load_dataset
import numpy as np
from .training_data import SYSTEM_PROMPT

FORMAT = SYSTEM_PROMPT.strip()

MATH = (
    "You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.\n"
    + FORMAT
)
COUNTDOWN = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. "
    "You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. "
    "After reasoning, provide only your final expression inside \\boxed{} tags without including an equals sign or the target number. "
    "For example: \\boxed{a + b * c}\n" + FORMAT
)


def records(task):
    if task == "countdown":
        path = Path(__file__).resolve().parents[2] / "data/countdown_cd3_test.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for i in np.random.choice(len(lines), 256, replace=False):
            row = json.loads(lines[i])
            numbers = [int(v) for v in row["input"].split(",")]
            target = int(row["output"])
            yield (
                str(i),
                f"{COUNTDOWN}\n\nNumbers: {numbers}\nTarget: {target}",
                (numbers, target),
            )
    elif task == "gsm8k":
        for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
            match = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", r["answer"])
            yield (
                str(i),
                MATH + "\n\n" + r["question"],
                float(match[1].replace(",", "")) if match else None,
            )
    elif task == "svamp":
        for i, r in enumerate(load_dataset("ChilleD/SVAMP", split="test")):
            yield str(i), MATH + "\n\n" + r["Body"] + r["Question"], r["Answer"]
    elif task == "humaneval":
        for r in load_dataset("openai/openai_humaneval", split="test"):
            yield (
                r["task_id"],
                "Complete the following Python function:\n" + r["prompt"],
                r,
            )
    elif task == "mbpp":
        for r in load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        ):
            tests = "\n".join(r["test_list"])
            prompt = (
                "You are a coding expert. You will be given a coding problem to solve. "
                "Write a Python function to solve the problem. Output only the function code wrapped in "
                "\x60\x60\x60python\x60\x60\x60 blocks. Do not include explanations or extra text.\n\n"
                + r["prompt"]
                + "\n\nYour code should pass the following tests:\n\n"
                + tests
            )
            yield str(r["task_id"]), prompt, r


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s


def numeric_answer(text):
    for value in re.findall(r"\\boxed{(.*?)}", text):
        try:
            return float(value.strip())
        except ValueError:
            numbers = re.findall(r"-?\d+\.?\d*", value)
            if numbers:
                return float(numbers[0])
    tag = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if tag:
        try:
            return float(tag[1].strip())
        except ValueError:
            numbers = re.findall(r"-?\d+\.?\d*", tag[1])
            if numbers:
                return float(numbers[-1])
    return None


def score(task, text, target):
    if task != "countdown":
        answer = numeric_answer(text)
        return answer is not None and target is not None and answer == float(target)
    numbers, goal = target
    try:
        expr = remove_boxed(last_boxed_only_string(text))
    except (TypeError, AssertionError):
        tag = re.search(r"<answer>(.*?)</answer>", text, re.S)
        expr = tag[1].strip() if tag else text
    expr = (
        expr.replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace(r"\div", "/")
        .strip()
    )
    equation = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", expr)
    if equation:
        expr = equation[1].strip()
    try:
        if not re.fullmatch(r"[\d.\s()+*/-]+", expr):
            return False
        if sorted(map(int, re.findall(r"\d+", expr))) != sorted(numbers):
            return False
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def value(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                operand = value(node.operand)
                return -operand if isinstance(node.op, ast.USub) else operand
            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                return operations[type(node.op)](value(node.left), value(node.right))
            raise ValueError("Unsupported expression")

        result = value(ast.parse(expr, mode="eval").body)
        return abs(result - goal) < 1e-5
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        return False


def code_solution(task, generated, record):
    match = re.search(
        r"\x60\x60\x60(?:python|py)?[ \t]*\r?\n(.*?)\x60\x60\x60",
        generated,
        re.S | re.I,
    )
    code = (match[1] if match else generated).rstrip()
    if task == "humaneval" and not re.search(
        r"(?m)^def\s+" + re.escape(record["entry_point"]) + r"\s*\(", code
    ):
        code = record["prompt"] + code
    return code
