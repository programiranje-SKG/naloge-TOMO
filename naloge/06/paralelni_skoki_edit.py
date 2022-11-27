with open(__file__, encoding='utf-8') as f:
    source = f.read()
exec(source[source.find("# =L=I=B=""R=A=R=Y=@="):])
problem = extract_problem(__file__)
Check.initialize(problem['parts'])

# =============================================================================
# Paralelni skoki
# =====================================================================@032883=
# V Planici so zgradili novo, dvojno skakalnico, s katero lahko izvajajo meddržavna tekmovanja v paralelnih skokih.
# Stvar je preprosta: skakalci se paroma odganjajo z dveh skakalnic, eni z ene, drugi z druge. Država, katere skakalec
# je skočil dlje, dobi točko; če sta skoka enako dolga, dobi vsaka država polovico točke. Zmaga država, ki dobi več točk.
# Napiši funkcijo z imenom `paralelni_skoki`, ki dobi seznama z dolžinami skokov. Če je zmagala prva država, naj funkcija vrne `1`, sicer `2`. Če sta
# državi izenačeni, naj funkcija ne vrne ničesar (`None`).
# 
# Primer:
# 
#      >>> paralelni_skoki([153, 141, 152, 160, 135], [148, 148, 148, 148, 148])
#      1
# 
# Finci so zmagali, ker so bili boljši v treh skokih, (fascinantno usklajeni) Danci pa le v dveh.
# =============================================================================
def paralelni_skoki(dolzine1, dolzine2):
    zmage1 = zmage2 = 0
    for skok1, skok2 in zip(dolzine1, dolzine2):
        if skok1 > skok2:
            zmage1 += 1
        elif skok2 > skok1:
            zmage2 += 1
        else:
            zmage1 += 0.5
            zmage2 += 0.5
    if zmage1 > zmage2:
        return 1
    elif zmage2 > zmage1:
        return 2

Check.part()
Check.equal('paralelni_skoki([153, 141, 152, 160, 135], [148, 148, 148, 148, 148])', 1)
Check.equal('paralelni_skoki([190, 159, 143, 183, 157], [143, 154, 169, 158, 149])', 1)
Check.equal('paralelni_skoki([179, 178, 145, 151, 146, 171], [147, 174, 163, 162, 169, 178])', 2)
Check.equal('paralelni_skoki([179, 178, 145, 151, 146, 179], [147, 174, 163, 162, 169, 178])', None)
Check.equal('paralelni_skoki([142, 160, 187, 182], [174, 172, 189, 172])', 2)
# Check.equal('multiply(6, 7)', 42)
# Check.equal('multiply(10, 10)', 100)
# Check.secret(multiply(100, 100))
# Check.secret(multiply(500, 123))


# # =====================================================================@000000=
# # This is a template for a new problem part. To create a new part, uncomment
# # the template and fill in your content.
# #
# # Define a function `multiply(x, y)` that returns the product of `x` and `y`.
# # For example:
# #
# #     >>> multiply(3, 7)
# #     21
# #     >>> multiply(6, 7)
# #     42
# # =============================================================================
#
# def multiply(x, y):
#     return x * y
#
# Check.part()
#
# Check.equal('multiply(3, 7)', 21)
# Check.equal('multiply(6, 7)', 42)
# Check.equal('multiply(10, 10)', 100)
# Check.secret(multiply(100, 100))
# Check.secret(multiply(500, 123))


# ===========================================================================@=
# Do not change this line or anything below it.
# =============================================================================


if __name__ == '__main__':
    _validate_current_file()

# =L=I=B=R=A=R=Y=@=

import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request

import io
import sys
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def extract_problem(filename):
    def strip_hashes(description):
        if description is None:
            return ''
        else:
            lines = description.strip().splitlines()
            return "\n".join(line[line.index('#')+2:] for line in lines)

    with open(filename, encoding='utf-8') as f:
        source = f.read()
    part_regex = re.compile(
        r'# ===+@(?P<part>\d+)=\s*\n'             # beginning of part header
        r'(?P<description>(\s*#( [^\n]*)?\n)+?)'  # description
        r'(\s*# ---+\s*\n'                        # optional beginning of template
        r'(?P<template>(\s*#( [^\n]*)?\n)*))?'    # solution template
        r'\s*# ===+\s*?\n'                        # end of part header
        r'(?P<solution>.*?)'                      # solution
        r'^Check\s*\.\s*part\s*\(\s*\)\s*?(?=\n)' # beginning of validation
        r'(?P<validation>.*?)'                    # validation
        r'(?=\n\s*(# )?# =+@)',                   # beginning of next part
        flags=re.DOTALL | re.MULTILINE
    )
    parts = [{
        'part': int(match.group('part')),
        'description': strip_hashes(match.group('description')),
        'solution': match.group('solution').strip(),
        'template': strip_hashes(match.group('template')),
        'validation': match.group('validation').strip(),
        'problem': 12302
    } for match in part_regex.finditer(source)]
    problem_match = re.search(
        r'^\s*# =+\s*\n'                          # beginning of header
        r'^\s*# (?P<title>[^\n]*)\n'              # title
        r'(?P<description>(^\s*#( [^\n]*)?\n)*?)' # description
        r'(?=\s*(# )?# =+@)',                     # beginning of first part
        source, flags=re.DOTALL | re.MULTILINE)
    return {
        'title': problem_match.group('title').strip(),
        'description': strip_hashes(problem_match.group('description')),
        'parts': parts,
        'id': 12302,
        'problem_set': 2539
    }

def _validate_current_file():
    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = '{0}.{1}'.format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_problem(problem, url, token):
        for part in problem['parts']:
            part['secret'] = [x for (x, _) in part['secret']]
            if part['part']:
                part['id'] = part['part']
            del part['part']
            del part['feedback']
            del part['valid']
        data = json.dumps(problem).encode('utf-8')
        headers = {
            'Authorization': token,
            'content-type': 'application/json'
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))

    Check.summarize()
    if all(part['valid'] for part in problem['parts']):
        print('The problem is correctly formulated.')
        if input('Should I save it on the server [yes/NO]') == 'yes':
            print('Saving problem to the server...', end="")
            try:
                url = 'https://www.projekt-tomo.si/api/problems/submit/'
                token = 'Token af2b966241aab0615e0c39a490d5543e12ff8245'
                response = submit_problem(problem, url, token)
                if 'update' in response:
                    print('Updating file... ', end="")
                    backup_filename = backup(__file__)
                    with open(__file__, 'w', encoding='utf-8') as f:
                        f.write(response['update'])
                    print('Previous file has been renamed to {0}.'.format(backup_filename))
                    print('If the file did not refresh in your editor, close and reopen it.')
            except urllib.error.URLError as response:
                message = json.loads(response.read().decode('utf-8'))
                print('\nAN ERROR OCCURED WHEN TRYING TO SAVE THE PROBLEM!')
                if message:
                    print('  ' + '\n  '.join(message.splitlines()))
                print('Please, try again.')
            else:
                print('Problem saved.')
        else:
            print('Problem was not saved.')
    else:
        print('The problem is not correctly formulated.')
