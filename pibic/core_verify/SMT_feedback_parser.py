import re

class FeedbackTrace:
    """
    Parses native ESBMC SMT logs to extract the mathematical counter-example trace
    isolating garbage output from strictly verified bug categories.
    """
    
    # Maps > 15 bug categories to Regex to identify what kind of trace was produced
    PATTERNS = {
        "array out of bounds": r"(?i)array bounds violated|array out of bounds",
        "arithmetic overflow": r"(?i)arithmetic overflow",
        "memory leak": r"(?i)memory leak|dereference failure",
        "division by zero": r"(?i)division by zero",
        "pointer dereference": r"(?i)pointer dereference|invalid pointer",
        "assertion failure": r"(?i)assertion.*failed|violated property",
        "nan/inf float detection": r"(?i)NaN or Inf",
        "loop unwinding failure": r"(?i)unwinding assertion",
        "solver abort": r"(?i)aborted|solver abort",
        "uninitialized variable reading": r"(?i)uninitialized",
        "double free": r"(?i)double free",
        "use after free": r"(?i)use after free|freed memory",
        "deadlock detection": r"(?i)deadlock",
        "data race": r"(?i)data race"
    }

    def __init__(self, raw_output, is_timeout_flag=False):
        self.raw_output = raw_output
        self.violations = []
        self.is_timeout = is_timeout_flag
        self.parse_violations()
        
    def parse_violations(self):
        if self.is_timeout:
            self.violations.append("target timeout")
            return
            
        for name, pattern in self.PATTERNS.items():
            if re.search(pattern, self.raw_output):
                self.violations.append(name)
                
    def extract_llm_prompt_context(self):
        """
        Calculates and strips the trace specifically so an LLM can consume it
        without becoming hallucinated by the GOTO symex conversion metadata.
        """
        if self.is_timeout:
            return "[TIMEOUT DE DETECÇÃO] - O Solver não convergiu no tempo adequado. O loop de análise pode ser infinito ou os bounds do FloatBV/FixedBV excederam limite de cálculo."

        if not self.violations:
            return "ESBMC Failed, mas nenhuma falha estrutural matemática mapeada foi identificada (Generic Error).\n" + "\n".join(self.raw_output.splitlines()[-20:])
            
        snippet = []
        capture = False
        
        for line in self.raw_output.splitlines():
            lc = line.lower()
            if not capture:
                if any(re.search(p, lc) for p in self.PATTERNS.values()):
                    capture = True
            
            if capture:
                snippet.append(line)
                if "VERIFICATION FAILED" in line:
                    break
        
        # If regex missed the start block, fallback to last relevant trace lines
        if not snippet:
             snippet = self.raw_output.splitlines()[-40:]
             
        return "\n".join(snippet)
