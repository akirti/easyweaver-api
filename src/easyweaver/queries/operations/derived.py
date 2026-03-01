import re

import polars as pl
import structlog

logger = structlog.get_logger()

_COL_REF_RE = re.compile(r"\{(\w[\w.]*)\}")
_ALLOWED_MATH_RE = re.compile(r"^[\d\s\+\-\*/\(\)\.\{\}\w]+$")


def apply_derived_columns(df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    """Apply derived column specifications to produce new computed columns."""
    for spec in specs:
        expr_type = spec["expression_type"]
        name = spec["name"]

        try:
            if expr_type == "concat":
                df = _apply_concat(df, spec, name)
            elif expr_type == "math":
                df = _apply_math(df, spec, name)
            elif expr_type == "date_part":
                df = _apply_date_part(df, spec, name)
            elif expr_type == "conditional":
                df = _apply_conditional(df, spec, name)
            elif expr_type == "literal":
                df = df.with_columns(pl.lit(spec.get("value")).alias(name))
            else:
                logger.warning("derived_unknown_type", expression_type=expr_type, name=name)
        except Exception:
            logger.warning("derived_column_failed", name=name, expression_type=expr_type, exc_info=True)

    return df


def _apply_concat(df: pl.DataFrame, spec: dict, name: str) -> pl.DataFrame:
    cols = spec.get("columns") or []
    sep = spec.get("separator", "")
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return df
    expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in valid], separator=sep)
    return df.with_columns(expr.alias(name))


def _apply_math(df: pl.DataFrame, spec: dict, name: str) -> pl.DataFrame:
    expression = spec.get("expression", "")
    if not expression:
        return df
    expr = _parse_math_expr(expression, df)
    if expr is None:
        return df
    return df.with_columns(expr.alias(name))


def _apply_date_part(df: pl.DataFrame, spec: dict, name: str) -> pl.DataFrame:
    col = spec.get("source_column", "")
    part = spec.get("part", "")
    if col not in df.columns:
        return df

    part_map = {
        "year": lambda c: pl.col(c).dt.year(),
        "month": lambda c: pl.col(c).dt.month(),
        "day": lambda c: pl.col(c).dt.day(),
        "hour": lambda c: pl.col(c).dt.hour(),
        "minute": lambda c: pl.col(c).dt.minute(),
        "second": lambda c: pl.col(c).dt.second(),
        "day_of_week": lambda c: pl.col(c).dt.weekday(),
        "quarter": lambda c: pl.col(c).dt.quarter(),
    }
    if part not in part_map:
        return df
    return df.with_columns(part_map[part](col).alias(name))


def _apply_conditional(df: pl.DataFrame, spec: dict, name: str) -> pl.DataFrame:
    col = spec.get("condition_column", "")
    op = spec.get("condition_operator", "")
    cond_val = spec.get("condition_value")
    then_val = spec.get("then_value")
    else_val = spec.get("else_value")

    if col not in df.columns:
        return df

    condition = _build_condition(col, op, cond_val)
    if condition is None:
        return df

    return df.with_columns(
        pl.when(condition).then(pl.lit(then_val)).otherwise(pl.lit(else_val)).alias(name)
    )


def _build_condition(col: str, op: str, value) -> pl.Expr | None:
    ops = {
        "eq": lambda: pl.col(col) == value,
        "neq": lambda: pl.col(col) != value,
        "gt": lambda: pl.col(col) > value,
        "lt": lambda: pl.col(col) < value,
        "gte": lambda: pl.col(col) >= value,
        "lte": lambda: pl.col(col) <= value,
        "is_null": lambda: pl.col(col).is_null(),
        "is_not_null": lambda: pl.col(col).is_not_null(),
    }
    fn = ops.get(op)
    return fn() if fn else None


def _parse_math_expr(expression: str, df: pl.DataFrame) -> pl.Expr | None:
    """Parse a simple math expression like '{col1} * {col2} + 10'.

    Only supports: +, -, *, /, column references in {braces}, numeric literals, parentheses.
    Does NOT use eval().
    """
    if not _ALLOWED_MATH_RE.match(expression):
        logger.warning("derived_math_invalid_chars", expression=expression)
        return None

    tokens = _tokenize_math(expression, df)
    if tokens is None:
        return None

    try:
        result, pos = _parse_additive(tokens, 0)
        if pos != len(tokens):
            return None
        return result
    except (IndexError, ValueError):
        return None


def _tokenize_math(expression: str, df: pl.DataFrame) -> list | None:
    """Tokenize a math expression into a list of (type, value) tuples."""
    tokens = []
    i = 0
    s = expression.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == '{':
            end = s.index('}', i + 1)
            col_name = s[i + 1:end]
            if col_name not in df.columns:
                logger.warning("derived_math_missing_column", column=col_name)
                return None
            tokens.append(("col", col_name))
            i = end + 1
        elif s[i] in "+-*/":
            tokens.append(("op", s[i]))
            i += 1
        elif s[i] == '(':
            tokens.append(("lparen", "("))
            i += 1
        elif s[i] == ')':
            tokens.append(("rparen", ")"))
            i += 1
        elif s[i].isdigit() or s[i] == '.':
            j = i
            has_dot = s[i] == '.'
            i += 1
            while i < len(s) and (s[i].isdigit() or (s[i] == '.' and not has_dot)):
                if s[i] == '.':
                    has_dot = True
                i += 1
            tokens.append(("num", float(s[j:i])))
        else:
            return None
    return tokens


def _parse_additive(tokens: list, pos: int) -> tuple[pl.Expr, int]:
    """Parse additive expressions: expr (+|-) expr"""
    left, pos = _parse_multiplicative(tokens, pos)
    while pos < len(tokens) and tokens[pos] == ("op", "+") or (
        pos < len(tokens) and tokens[pos] == ("op", "-")
    ):
        op = tokens[pos][1]
        pos += 1
        right, pos = _parse_multiplicative(tokens, pos)
        if op == "+":
            left = left + right
        else:
            left = left - right
    return left, pos


def _parse_multiplicative(tokens: list, pos: int) -> tuple[pl.Expr, int]:
    """Parse multiplicative expressions: expr (*|/) expr"""
    left, pos = _parse_primary(tokens, pos)
    while pos < len(tokens) and (
        tokens[pos] == ("op", "*") or tokens[pos] == ("op", "/")
    ):
        op = tokens[pos][1]
        pos += 1
        right, pos = _parse_primary(tokens, pos)
        if op == "*":
            left = left * right
        else:
            left = left / right
    return left, pos


def _parse_primary(tokens: list, pos: int) -> tuple[pl.Expr, int]:
    """Parse primary expressions: column, number, or parenthesized expression."""
    token = tokens[pos]
    if token[0] == "col":
        return pl.col(token[1]), pos + 1
    elif token[0] == "num":
        return pl.lit(token[1]), pos + 1
    elif token[0] == "lparen":
        expr, pos = _parse_additive(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos][0] != "rparen":
            raise ValueError("Unmatched parenthesis")
        return expr, pos + 1
    raise ValueError(f"Unexpected token: {token}")
