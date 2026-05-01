from dataclasses import dataclass
import re
import csv
from pathlib import Path
from time import perf_counter


# Logical constants: true and false
# Top = True, Bottom = False
@dataclass(frozen=True)
class Top:
    def __str__(self):
        return "T"


@dataclass(frozen=True)
class Bottom:
    def __str__(self):
        return "F"


# Atomic predicate, e.g. P(a), Q(x), R(x,y)
@dataclass(frozen=True)
class Atom:
    name: str
    args: tuple

    def __str__(self):
        if len(self.args) == 0:
            return self.name
        return self.name + "(" + ",".join(self.args) + ")"


# Negation, e.g. not P(a)
@dataclass(frozen=True)
class Not:
    formula: object

    def __str__(self):
        return "not " + str(self.formula)


# Conjunction, e.g. P(a) and Q(a)
@dataclass(frozen=True)
class And:
    left: object
    right: object

    def __str__(self):
        return "(" + str(self.left) + " and " + str(self.right) + ")"


# Disjunction, e.g. P(a) or Q(a)
@dataclass(frozen=True)
class Or:
    left: object
    right: object

    def __str__(self):
        return "(" + str(self.left) + " or " + str(self.right) + ")"


# Implication, e.g. P(a) -> Q(a)
@dataclass(frozen=True)
class Imp:
    left: object
    right: object

    def __str__(self):
        return "(" + str(self.left) + " -> " + str(self.right) + ")"


# Universal quantifier, e.g. forall x. P(x)
@dataclass(frozen=True)
class Forall:
    variable: str
    formula: object

    def __str__(self):
        return "forall " + self.variable + ". " + str(self.formula)


# Existential quantifier, e.g. exists x. P(x)
@dataclass(frozen=True)
class Exists:
    variable: str
    formula: object

    def __str__(self):
        return "exists " + self.variable + ". " + str(self.formula)

# Turns a formula string into separate tokens
def tokenize(text):
    tokens = []
    pos = 0

    pattern = r"\s*(->|[(),.]|[A-Za-z_][A-Za-z0-9_]*)"

    while pos < len(text):
        match = re.match(pattern, text[pos:])

        if not match:
            raise ValueError("Could not read formula near: " + text[pos:])

        token = match.group(1)
        tokens.append(token)
        pos += match.end()

    return tokens


# Parser for the formula syntax used in the datasets
class Parser:
    def __init__(self, text):
        self.tokens = tokenize(text)
        self.pos = 0

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self, expected=None):
        token = self.current()

        if token is None:
            raise ValueError("Unexpected end of formula")

        if expected is not None and token != expected:
            raise ValueError("Expected " + expected + " but got " + token)

        self.pos += 1
        return token

    def parse(self):
        formula = self.parse_implication()

        if self.current() is not None:
            raise ValueError("Unexpected token: " + self.current())

        return formula

    # Lowest priority: implication
    def parse_implication(self):
        left = self.parse_or()

        if self.current() == "->":
            self.eat("->")
            right = self.parse_implication()
            return Imp(left, right)

        return left

    # Next priority: or
    def parse_or(self):
        left = self.parse_and()

        while self.current() == "or":
            self.eat("or")
            right = self.parse_and()
            left = Or(left, right)

        return left

    # Next priority: and
    def parse_and(self):
        left = self.parse_unary()

        while self.current() == "and":
            self.eat("and")
            right = self.parse_unary()
            left = And(left, right)

        return left

    # Highest priority: not, quantifiers, brackets, atoms
    def parse_unary(self):
        token = self.current()

        if token == "not":
            self.eat("not")
            return Not(self.parse_unary())

        if token == "forall":
            self.eat("forall")
            variable = self.eat()
            self.eat(".")
            return Forall(variable, self.parse_unary())

        if token == "exists":
            self.eat("exists")
            variable = self.eat()
            self.eat(".")
            return Exists(variable, self.parse_unary())

        if token == "(":
            self.eat("(")
            formula = self.parse_implication()
            self.eat(")")
            return formula

        return self.parse_atom()

    # Predicates and constants, e.g. P(a), R(x,y), T, F
    def parse_atom(self):
        name = self.eat()

        if name == "T":
            return Top()

        if name == "F":
            return Bottom()

        args = []

        if self.current() == "(":
            self.eat("(")

            if self.current() != ")":
                args.append(self.eat())

                while self.current() == ",":
                    self.eat(",")
                    args.append(self.eat())

            self.eat(")")

        return Atom(name, tuple(args))


# Helper function so the rest of the code can just call parse_formula(...)
def parse_formula(text):
    parser = Parser(text)
    return parser.parse()

# Applies one round of simplification rules
def simplify_step(formula):

    # Basic formulae cannot be simplified
    if isinstance(formula, Top):
        return formula

    if isinstance(formula, Bottom):
        return formula

    if isinstance(formula, Atom):
        return formula

    # Double negation: not not A -> A
    if isinstance(formula, Not):
        inside = simplify_step(formula.formula)

        if isinstance(inside, Not):
            return inside.formula

        return Not(inside)

    # AND rules
    if isinstance(formula, And):
        left = simplify_step(formula.left)
        right = simplify_step(formula.right)

        # A and F -> F
        # F and A -> F
        if isinstance(left, Bottom) or isinstance(right, Bottom):
            return Bottom()

        # T and A -> A
        if isinstance(left, Top):
            return right

        # A and T -> A
        if isinstance(right, Top):
            return left

        return And(left, right)

    # OR rules
    if isinstance(formula, Or):
        left = simplify_step(formula.left)
        right = simplify_step(formula.right)

        # A or T -> T
        # T or A -> T
        if isinstance(left, Top) or isinstance(right, Top):
            return Top()

        # F or A -> A
        if isinstance(left, Bottom):
            return right

        # A or F -> A
        if isinstance(right, Bottom):
            return left

        return Or(left, right)

    # Implication rules
    if isinstance(formula, Imp):
        left = simplify_step(formula.left)
        right = simplify_step(formula.right)

        # A -> A -> T
        if left == right:
            return Top()

        # A -> T -> T
        if isinstance(right, Top):
            return Top()

        # F -> A -> T
        if isinstance(left, Bottom):
            return Top()

        # T -> A -> A
        if isinstance(left, Top):
            return right

        return Imp(left, right)

    # Universal quantifier rule
    if isinstance(formula, Forall):
        inside = simplify_step(formula.formula)

        # forall x. T -> T
        if isinstance(inside, Top):
            return Top()

        return Forall(formula.variable, inside)

    # Existential quantifier rule
    if isinstance(formula, Exists):
        inside = simplify_step(formula.formula)

        # exists x. F -> F
        if isinstance(inside, Bottom):
            return Bottom()

        return Exists(formula.variable, inside)

    return formula

# Keeps simplifying until nothing changes
def simplify(formula):
    old_formula = formula
    new_formula = simplify_step(old_formula)

    while new_formula != old_formula:
        old_formula = new_formula
        new_formula = simplify_step(old_formula)

    return new_formula

# A sequent has formulae on the left and right side
@dataclass(frozen=True)
class Sequent:
    left: frozenset
    right: frozenset

    def __str__(self):
        left_text = ", ".join(sorted(str(f) for f in self.left))
        right_text = ", ".join(sorted(str(f) for f in self.right))

        if left_text == "":
            left_text = " "

        if right_text == "":
            right_text = " "

        return left_text + " |- " + right_text


# A branch stores the current sequent and the next fresh term number
@dataclass(frozen=True)
class Branch:
    sequent: Sequent
    next_fresh: int


# Replaces a variable with a term inside a formula
def substitute(formula, variable, term):

    if isinstance(formula, Atom):
        new_args = []

        for arg in formula.args:
            if arg == variable:
                new_args.append(term)
            else:
                new_args.append(arg)

        return Atom(formula.name, tuple(new_args))

    if isinstance(formula, Not):
        return Not(substitute(formula.formula, variable, term))

    if isinstance(formula, And):
        return And(
            substitute(formula.left, variable, term),
            substitute(formula.right, variable, term)
        )

    if isinstance(formula, Or):
        return Or(
            substitute(formula.left, variable, term),
            substitute(formula.right, variable, term)
        )

    if isinstance(formula, Imp):
        return Imp(
            substitute(formula.left, variable, term),
            substitute(formula.right, variable, term)
        )

    if isinstance(formula, Forall):
        if formula.variable == variable:
            return formula

        return Forall(
            formula.variable,
            substitute(formula.formula, variable, term)
        )

    if isinstance(formula, Exists):
        if formula.variable == variable:
            return formula

        return Exists(
            formula.variable,
            substitute(formula.formula, variable, term)
        )

    return formula


# Finds constants/terms already used in a formula
def collect_terms(formula):
    terms = set()

    if isinstance(formula, Atom):
        for arg in formula.args:
            if arg not in ["x", "y", "z"]:
                terms.add(arg)

    elif isinstance(formula, Not):
        terms.update(collect_terms(formula.formula))

    elif isinstance(formula, And) or isinstance(formula, Or) or isinstance(formula, Imp):
        terms.update(collect_terms(formula.left))
        terms.update(collect_terms(formula.right))

    elif isinstance(formula, Forall) or isinstance(formula, Exists):
        terms.update(collect_terms(formula.formula))

    return terms


# Finds all terms currently appearing in a sequent
def collect_sequent_terms(sequent):
    terms = set()

    for formula in sequent.left:
        terms.update(collect_terms(formula))

    for formula in sequent.right:
        terms.update(collect_terms(formula))

    # Use a default constant if there are no existing terms
    if len(terms) == 0:
        terms.add("a")

    return sorted(terms)


# Checks whether a branch is closed
def is_closed(sequent):

    # id rule: same formula on both sides
    for formula in sequent.left:
        if formula in sequent.right:
            return True

    # F on the left closes the branch
    for formula in sequent.left:
        if isinstance(formula, Bottom):
            return True

    # T on the right closes the branch
    for formula in sequent.right:
        if isinstance(formula, Top):
            return True

    return False


# Helper for making a new branch
def make_branch(sequent, branch, next_fresh=None):
    if next_fresh is None:
        next_fresh = branch.next_fresh

    return Branch(sequent, next_fresh)


# Applies one backwards proof-search rule
def expand_branch(branch, fresh_limit):
    sequent = branch.sequent

    left_side = sorted(sequent.left, key=str)
    right_side = sorted(sequent.right, key=str)

    # Non-branching rules on the left side
    for formula in left_side:

        # AND left: A and B on the left becomes A, B on the left
        if isinstance(formula, And):
            new_left = set(sequent.left)
            new_left.remove(formula)
            new_left.add(formula.left)
            new_left.add(formula.right)

            new_sequent = Sequent(frozenset(new_left), sequent.right)
            return [make_branch(new_sequent, branch)]

        # NOT left: not A on the left moves A to the right
        if isinstance(formula, Not):
            new_left = set(sequent.left)
            new_right = set(sequent.right)

            new_left.remove(formula)
            new_right.add(formula.formula)

            new_sequent = Sequent(frozenset(new_left), frozenset(new_right))
            return [make_branch(new_sequent, branch)]

        # EXISTS left: create a fresh term
        if isinstance(formula, Exists):
            if branch.next_fresh >= fresh_limit:
                continue

            fresh = "c" + str(branch.next_fresh)

            new_left = set(sequent.left)
            new_left.remove(formula)
            new_left.add(substitute(formula.formula, formula.variable, fresh))

            new_sequent = Sequent(frozenset(new_left), sequent.right)
            return [make_branch(new_sequent, branch, branch.next_fresh + 1)]

    # Non-branching rules on the right side
    for formula in right_side:

        # OR right: A or B on the right becomes A, B on the right
        if isinstance(formula, Or):
            new_right = set(sequent.right)
            new_right.remove(formula)
            new_right.add(formula.left)
            new_right.add(formula.right)

            new_sequent = Sequent(sequent.left, frozenset(new_right))
            return [make_branch(new_sequent, branch)]

        # Implication right: A -> B on the right moves A to left and B stays right
        if isinstance(formula, Imp):
            new_left = set(sequent.left)
            new_right = set(sequent.right)

            new_right.remove(formula)
            new_left.add(formula.left)
            new_right.add(formula.right)

            new_sequent = Sequent(frozenset(new_left), frozenset(new_right))
            return [make_branch(new_sequent, branch)]

        # NOT right: not A on the right moves A to the left
        if isinstance(formula, Not):
            new_left = set(sequent.left)
            new_right = set(sequent.right)

            new_right.remove(formula)
            new_left.add(formula.formula)

            new_sequent = Sequent(frozenset(new_left), frozenset(new_right))
            return [make_branch(new_sequent, branch)]

        # FORALL right: create a fresh term
        if isinstance(formula, Forall):
            if branch.next_fresh >= fresh_limit:
                continue

            fresh = "c" + str(branch.next_fresh)

            new_right = set(sequent.right)
            new_right.remove(formula)
            new_right.add(substitute(formula.formula, formula.variable, fresh))

            new_sequent = Sequent(sequent.left, frozenset(new_right))
            return [make_branch(new_sequent, branch, branch.next_fresh + 1)]

    # Branching rule: AND right
    for formula in right_side:
        if isinstance(formula, And):
            right_one = set(sequent.right)
            right_two = set(sequent.right)

            right_one.remove(formula)
            right_two.remove(formula)

            right_one.add(formula.left)
            right_two.add(formula.right)

            sequent_one = Sequent(sequent.left, frozenset(right_one))
            sequent_two = Sequent(sequent.left, frozenset(right_two))

            return [
                make_branch(sequent_one, branch),
                make_branch(sequent_two, branch)
            ]

    # Branching rules on the left side
    for formula in left_side:

        # OR left
        if isinstance(formula, Or):
            left_one = set(sequent.left)
            left_two = set(sequent.left)

            left_one.remove(formula)
            left_two.remove(formula)

            left_one.add(formula.left)
            left_two.add(formula.right)

            sequent_one = Sequent(frozenset(left_one), sequent.right)
            sequent_two = Sequent(frozenset(left_two), sequent.right)

            return [
                make_branch(sequent_one, branch),
                make_branch(sequent_two, branch)
            ]

        # Implication left
        if isinstance(formula, Imp):
            left_one = set(sequent.left)
            right_one = set(sequent.right)

            left_two = set(sequent.left)
            right_two = set(sequent.right)

            left_one.remove(formula)
            left_two.remove(formula)

            right_one.add(formula.left)
            left_two.add(formula.right)

            sequent_one = Sequent(frozenset(left_one), frozenset(right_one))
            sequent_two = Sequent(frozenset(left_two), frozenset(right_two))

            return [
                make_branch(sequent_one, branch),
                make_branch(sequent_two, branch)
            ]

    # FORALL left and EXISTS right use existing terms first, then fresh terms
    terms = collect_sequent_terms(sequent)

    # FORALL left
    for formula in left_side:
        if isinstance(formula, Forall):

            for term in terms:
                instance = substitute(formula.formula, formula.variable, term)

                if instance not in sequent.left:
                    new_left = set(sequent.left)
                    new_left.add(instance)

                    new_sequent = Sequent(frozenset(new_left), sequent.right)
                    return [make_branch(new_sequent, branch)]

            if branch.next_fresh < fresh_limit:
                fresh = "c" + str(branch.next_fresh)
                instance = substitute(formula.formula, formula.variable, fresh)

                new_left = set(sequent.left)
                new_left.add(instance)

                new_sequent = Sequent(frozenset(new_left), sequent.right)
                return [make_branch(new_sequent, branch, branch.next_fresh + 1)]

    # EXISTS right
    for formula in right_side:
        if isinstance(formula, Exists):

            for term in terms:
                instance = substitute(formula.formula, formula.variable, term)

                if instance not in sequent.right:
                    new_right = set(sequent.right)
                    new_right.add(instance)

                    new_sequent = Sequent(sequent.left, frozenset(new_right))
                    return [make_branch(new_sequent, branch)]

            if branch.next_fresh < fresh_limit:
                fresh = "c" + str(branch.next_fresh)
                instance = substitute(formula.formula, formula.variable, fresh)

                new_right = set(sequent.right)
                new_right.add(instance)

                new_sequent = Sequent(sequent.left, frozenset(new_right))
                return [make_branch(new_sequent, branch, branch.next_fresh + 1)]

    # No rule could be applied
    return []


# Baseline implementation of Algorithm 2
def prove_algorithm2(formula, max_steps=5000, fresh_limit=6):
    start_time = perf_counter()

    starting_sequent = Sequent(
        frozenset(),
        frozenset([formula])
    )

    open_branches = [
        Branch(starting_sequent, 0)
    ]

    steps = 0
    max_open_branches = 1

    while len(open_branches) > 0 and steps < max_steps:
        branch = open_branches.pop()
        sequent = branch.sequent

        steps += 1
        max_open_branches = max(max_open_branches, len(open_branches) + 1)

        if is_closed(sequent):
            continue

        new_branches = expand_branch(branch, fresh_limit)

        if len(new_branches) == 0:
            runtime = (perf_counter() - start_time) * 1000

            return {
                "proved": False,
                "status": "open branch",
                "steps": steps,
                "max_open_branches": max_open_branches,
                "runtime_ms": round(runtime, 3)
            }

        for new_branch in new_branches:
            open_branches.append(new_branch)

    runtime = (perf_counter() - start_time) * 1000

    if len(open_branches) == 0:
        status = "proved"
        proved = True
    else:
        status = "step limit"
        proved = False

    return {
        "proved": proved,
        "status": status,
        "steps": steps,
        "max_open_branches": max_open_branches,
        "runtime_ms": round(runtime, 3)
    }

# Improved method: simplify the formula before running Algorithm 2
def prove_improved(formula, max_steps=5000, fresh_limit=6):
    simplified_formula = simplify(formula)

    result = prove_algorithm2(
        simplified_formula,
        max_steps=max_steps,
        fresh_limit=fresh_limit
    )

    result["simplified_formula"] = str(simplified_formula)

    return result
# Quick tests for the baseline Algorithm 2 implementation

# Runs both methods on the same formula
def compare_methods(text):
    formula = parse_formula(text)
    simplified_formula = simplify(formula)

    baseline_result = prove_algorithm2(formula)
    improved_result = prove_improved(formula)

    print("Formula:   ", text)
    print("Parsed:    ", formula)
    print("Simplified:", simplified_formula)
    print()
    print("Baseline result:", baseline_result)
    print("Improved result:", improved_result)
    print("-" * 70)

# Reads formulas from a dataset text file
def read_formula_file(file_path):
    formulas = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        text = line.strip()

        # Skip blank lines and comments
        if text == "" or text.startswith("#"):
            continue

        formulas.append(text)

    return formulas


# Runs baseline and improved methods on all dataset files
def run_experiments():
    project_folder = Path(__file__).resolve().parent.parent
    dataset_folder = project_folder / "datasets"
    results_folder = project_folder / "results"

    # Create results folder if it does not exist
    results_folder.mkdir(exist_ok=True)

    dataset_files = [
        "textbook_style.txt",
        "simplification_heavy.txt",
        "mixed_formulae.txt"
    ]

    output_file = results_folder / "results.csv"

    rows = []

    for dataset_name in dataset_files:
        file_path = dataset_folder / dataset_name
        formulas = read_formula_file(file_path)

        print("Running dataset:", dataset_name)

        for text in formulas:
            formula = parse_formula(text)
            simplified_formula = simplify(formula)

            baseline_result = prove_algorithm2(formula)
            improved_result = prove_improved(formula)

            row = {
                "dataset": dataset_name,
                "formula": text,
                "parsed_formula": str(formula),
                "simplified_formula": str(simplified_formula),

                "baseline_proved": baseline_result["proved"],
                "baseline_status": baseline_result["status"],
                "baseline_steps": baseline_result["steps"],
                "baseline_max_open_branches": baseline_result["max_open_branches"],
                "baseline_runtime_ms": baseline_result["runtime_ms"],

                "improved_proved": improved_result["proved"],
                "improved_status": improved_result["status"],
                "improved_steps": improved_result["steps"],
                "improved_max_open_branches": improved_result["max_open_branches"],
                "improved_runtime_ms": improved_result["runtime_ms"]
            }

            rows.append(row)

    # Save results to CSV
    with open(output_file, "w", newline="") as file:
        fieldnames = [
            "dataset",
            "formula",
            "parsed_formula",
            "simplified_formula",

            "baseline_proved",
            "baseline_status",
            "baseline_steps",
            "baseline_max_open_branches",
            "baseline_runtime_ms",

            "improved_proved",
            "improved_status",
            "improved_steps",
            "improved_max_open_branches",
            "improved_runtime_ms"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Experiments finished.")
    print("Results saved to:", output_file)
    print("Total formulas tested:", len(rows))

# Run the full experiment
if __name__ == "__main__":
    run_experiments()