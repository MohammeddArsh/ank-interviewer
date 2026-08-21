"""InterviewSession — the state machine that runs a natural, sectioned interview.

Flow: greeting → sections (each: question → answer → follow-up → …) →
section transitions → closing → done.
"""

import random
import re
import threading
import time

from brain.llm import complete
from config import FOLLOW_UPS_PER_QUESTION, MAX_QUESTIONS
from interview import prompts
from interview.evaluator import evaluate as _evaluate
from interview.plan import generate_plan

DEFAULT_INTERVIEWER = {"name": "Alex", "role": "Recruiter"}

_TURN_RETRIES = 3

_FOLLOW_UP_INS = (
    "The candidate has just answered the current question. Ask ONE concise follow-up that "
    "probes deeper into their answer — invite a concrete example, challenge an assumption, "
    "or ask them to elaborate on something they skimmed over. Do not move to a new topic."
)
_BRIDGE_INS = (
    "Briefly and naturally acknowledge the candidate's answer, then ask the NEXT planned "
    "question shown above. Keep it short — a real interviewer wouldn't repeat the answer back."
)
_TRANSITION_INS = (
    "The candidate has finished this section. Give a short, natural transition that "
    "acknowledges their answer, previews the next section, then asks its first question. "
    "Sound like a real interviewer smoothly moving between topics."
)
_CLOSING_OFFER_INS = (
    "The interview is complete. Thank the candidate warmly, note it was a pleasure, and ask "
    "whether they have any questions about the role or the team. End your line with that question."
)
_CLOSING_NO_QUESTIONS_INS = (
    "The candidate has no questions. Deliver a warm, final closing — thank them again and wish "
    "them well with the next steps. End the interview."
)
_CLOSING_HAS_QUESTIONS_INS = (
    "The candidate has questions. Answer graciously and briefly, note that any company-specific "
    "details are best confirmed with the hiring team, then wrap up warmly and end the interview."
)

_NO_QUESTION_PHRASES = {
    "no", "nope", "nah", "none", "nothing", "not really", "no questions",
    "that's all", "thats all", "that's it", "thats it", "that is it",
    "im good", "i'm good", "no thank you", "not that i can think of",
}

# Phrases free reasoning models use when they leak their internal thinking
# instead of just speaking as the interviewer.
_META_MARKERS = (
    "the user wants me to act as",
    "the user asked me to act as",
    "i need to ask a follow-up",
    "i need to ask a follow up",
    "i need to transition",
    "let me re-read",
    "looking at the interview progress",
    "there's a contradiction",
    "there is a contradiction",
    "the instruction says",
    "the instructions",
    "the candidate has finished",
    "now move to the next section",
    "question 1 of 2",
    "question 1 of",
    "based on the candidate",
)


def _is_meta_rambling(text: str) -> bool:
    """True when a model reply narrates its reasoning instead of speaking."""
    low = " ".join((text or "").lower().split())
    return any(m in low for m in _META_MARKERS)


def _candidate_has_questions(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    if t in _NO_QUESTION_PHRASES:
        return False
    if re.match(r"^(no|nope|nah|none|nothing|not really|no questions)", t):
        return False
    return True


class InterviewSession:
    def __init__(self, job_description: str, resume_text: str, interviewer: dict = None):
        self.job_description = job_description
        self.resume_text = resume_text
        self.interviewer = interviewer or DEFAULT_INTERVIEWER
        self.plan = None
        self.sections = []
        self.section_idx = 0
        self.question_idx = 0
        self.follow_ups_used = 0
        self.answered_count = 0
        self.total_questions = 0
        self.current_question = None
        self.phase = "setup"
        self.transcript = []
        self.evaluation = None
        self.error = None

    # ---- lifecycle ---------------------------------------------------------

    def prepare(self) -> dict:
        """Build the interview plan without speaking. Returns a plan summary."""
        if self.plan is None:
            self.plan = generate_plan(self.job_description, self.resume_text, MAX_QUESTIONS)
        self.sections = self.plan["sections"]
        self.total_questions = sum(len(s["questions"]) for s in self.sections)
        self.section_idx = 0
        self.question_idx = 0
        self.phase = "ready"
        return {
            "sections": [s["title"] for s in self.sections],
            "total_questions": self.total_questions,
            "interviewer": self.interviewer,
            "state": self.build_state(),
        }

    def begin(self) -> dict:
        """Deliver the greeting + warm-up question. Only valid after prepare()."""
        if self.phase != "ready":
            raise ValueError("The interview is not ready to begin yet.")

        opening = (self.plan.get("greeting") or "").strip()
        warmup = self.current_question = self._current_section()["questions"][0]
        opener = f"{opening} {warmup}".strip()
        self._append_interviewer(opener)
        self.phase = "answering_main"
        return {"utterance": opener, "state": self.build_state()}

    def start(self) -> dict:
        """Legacy one-shot: build the plan and return the spoken opener."""
        self.prepare()
        return self.begin()

    def handle_answer(self, candidate_text: str, is_skip: bool = False) -> dict:
        """Record an answer (or skip) and return the interviewer's next utterance + state."""
        if self.phase == "done":
            return {"utterance": "", "state": self.build_state(), "done": True,
                    "evaluation": self.evaluation}

        self._append_candidate(candidate_text if not is_skip else "[Question skipped]")

        if self.phase == "answering_main":
            self.follow_ups_used = 0
            if FOLLOW_UPS_PER_QUESTION > 0:
                self.phase = "answering_followup"
                utterance = self._generate(_FOLLOW_UP_INS)
                self._append_interviewer(utterance)
                return self._respond(utterance)
            utterance = self._advance_question()
            return self._respond(utterance)

        if self.phase == "answering_followup":
            self.follow_ups_used += 1
            if self.follow_ups_used < FOLLOW_UPS_PER_QUESTION:
                utterance = self._generate(_FOLLOW_UP_INS)
                self._append_interviewer(utterance)
                return self._respond(utterance)
            utterance = self._advance_question()
            return self._respond(utterance)

        if self.phase == "closing":
            if is_skip or not _candidate_has_questions(candidate_text):
                utterance = self._generate(_CLOSING_NO_QUESTIONS_INS)
            else:
                utterance = self._generate(_CLOSING_HAS_QUESTIONS_INS)
            self._append_interviewer(utterance)
            self.phase = "done"
            self.evaluation = self._build_evaluation()
            return self._respond(utterance, done=True)

        return {"utterance": "", "state": self.build_state()}

    def end_early(self) -> dict:
        """End the interview immediately. Evaluation is computed in the background."""
        if self.phase != "done":
            self.phase = "done"
            self._start_evaluation()
        return {"state": self.build_state(), "done": True}

    def _start_evaluation(self):
        """Run evaluation on a daemon thread so /interview/end returns instantly."""
        if getattr(self, "_eval_thread", None) and self._eval_thread.is_alive():
            return
        self.evaluation = None
        self._eval_done = False

        def work():
            try:
                self.evaluation = self._build_evaluation()
            finally:
                self._eval_done = True

        self._eval_thread = threading.Thread(target=work, daemon=True)
        self._eval_thread.start()

    # ---- progression -------------------------------------------------------

    def _advance_question(self) -> str:
        """Move past the current question; return the next interviewer utterance."""
        self.answered_count += 1
        self.follow_ups_used = 0
        self.question_idx += 1

        section = self._current_section()
        if section and self.question_idx < len(section["questions"]):
            self.current_question = section["questions"][self.question_idx]
            self.phase = "answering_main"
            return self._generate(_BRIDGE_INS)

        # Finished this section — move to the next one.
        finished = self._current_section()
        self.section_idx += 1
        self.question_idx = 0
        section = self._current_section()
        if section:
            self.current_question = section["questions"][0]
            self.phase = "answering_main"
            progress = (
                f"Sections: {', '.join(s['title'] for s in self.sections)}. "
                f"The candidate has just finished section '{finished['title']}' "
                f"({finished.get('focus', '')}). Now move to the next section "
                f"'{section['title']}' ({section.get('focus', '')}), question 1 of "
                f"{len(section['questions'])} in this section. "
                f"{self.answered_count} of {self.total_questions} questions answered."
            )
            return self._generate(_TRANSITION_INS, progress=progress, upcoming=section["questions"][0])

        # All sections done — close the interview.
        self.current_question = None
        self.phase = "closing"
        return self._generate(_CLOSING_OFFER_INS)

    # ---- helpers -----------------------------------------------------------

    def _current_section(self):
        if 0 <= self.section_idx < len(self.sections):
            return self.sections[self.section_idx]
        return None

    def _append_interviewer(self, text: str):
        self.transcript.append({"role": "interviewer", "text": text})

    def _append_candidate(self, text: str):
        self.transcript.append({"role": "candidate", "text": text})

    def _respond(self, utterance: str, done: bool = False) -> dict:
        resp = {"utterance": utterance, "state": self.build_state()}
        if done:
            resp["done"] = True
            resp["evaluation"] = self.evaluation
        return resp

    def _generate(self, instruction: str, progress: str = None, upcoming: str = None) -> str:
        """Generate the interviewer's next spoken line via the LLM.

        Retries a few times (free-tier models are flaky); if every attempt
        fails, returns a natural instruction-aware fallback line instead of
        degrading the conversation. Outputs that reveal the model's inner
        reasoning (meta-narration) are treated as failures and retried.
        `progress`/`upcoming` may override the auto-derived context (used for
        coherent section transitions).
        """
        progress = progress or self._progress_text()
        upcoming = upcoming or self._upcoming_question_text()
        transcript = self._transcript_text()

        messages = [
            {"role": "system", "content": prompts.interviewer_system(self.interviewer)},
            {"role": "user", "content": prompts.turn_prompt(
                self.job_description, self.resume_text, progress, upcoming, transcript, instruction
            )},
        ]
        for attempt in range(_TURN_RETRIES):
            try:
                text, _ = complete(messages, temperature=0.6, max_tokens=300)
                text = (text or "").strip()
                if text and not _is_meta_rambling(text):
                    return text
                if text:
                    print(f"[LLM] turn {attempt + 1}/{_TURN_RETRIES} was meta-reasoning; retrying...")
                raise RuntimeError("empty or meta-reasoning output")
            except Exception as e:  # network / rate-limit / bad output fallback
                self.error = str(e)
                if attempt < _TURN_RETRIES - 1:
                    print(f"[LLM] turn failed on attempt {attempt + 1}/{_TURN_RETRIES}: {e}; retrying...")
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))
        print(f"[LLM] all turn attempts failed ({self.error}); using fallback line")
        return self._fallback_utterance(instruction, upcoming)

    def _fallback_utterance(self, instruction: str, upcoming: str) -> str:
        """Natural spoken line used when the LLM is unavailable."""
        if instruction == _FOLLOW_UP_INS:
            return "Thanks for sharing that. Could you go into a little more detail on that point?"
        if instruction == _BRIDGE_INS:
            return f"Thanks for that. Let's move to the next question. {upcoming}".strip()
        if instruction == _TRANSITION_INS:
            return f"Great. Let's move to the next section. {upcoming}".strip()
        if instruction == _CLOSING_OFFER_INS:
            return (
                "That brings us to the end of the interview — thank you so much for your time. "
                "Do you have any questions about the role or the team?"
            )
        if instruction == _CLOSING_NO_QUESTIONS_INS:
            return (
                "Thank you for your time today — it was a pleasure speaking with you. "
                "I wish you all the best with the next steps."
            )
        if instruction == _CLOSING_HAS_QUESTIONS_INS:
            return (
                "Thanks for your questions — the hiring team can confirm the finer details. "
                "It was great meeting you; best of luck with next steps."
            )
        if upcoming:
            return f"Let's keep going. {upcoming}".strip()
        return "Thank you — that brings us to the end of the interview."

    def _progress_text(self) -> str:
        section = self._current_section()
        if section:
            total = self.total_questions
            return (
                f"Sections: {', '.join(s['title'] for s in self.sections)}. "
                f"Currently in section '{section['title']}' ({section.get('focus', '')}), "
                f"question {self.question_idx + 1} of {len(section['questions'])} in this section. "
                f"{self.answered_count} of {total} questions fully answered."
            )
        return f"All {self.total_questions} questions have been asked."

    def _upcoming_question_text(self) -> str:
        if self.phase == "answering_followup":
            return f"The current question was: {self.current_question or ''}"
        section = self._current_section()
        if section and self.question_idx < len(section["questions"]):
            return section["questions"][self.question_idx]
        if section and self.question_idx == len(section["questions"]):
            return "(none — this is the last question of the section)"
        return "(none — interview is ending)"

    def _transcript_text(self, limit: int = 6000) -> str:
        lines = [f"{'Interviewer' if t['role'] == 'interviewer' else 'Candidate'}: {t['text']}"
                 for t in self.transcript]
        text = "\n".join(lines)
        return text[-limit:] if len(text) > limit else text

    def _build_evaluation(self) -> dict:
        try:
            return _evaluate(self.job_description, self.resume_text, self.transcript)
        except Exception as e:
            self.error = str(e)
            return {"score": 0, "strengths": [], "improvements": [], "verdict": ""}

    # ---- state -------------------------------------------------------------

    def build_state(self) -> dict:
        section = self._current_section()
        return {
            "phase": self.phase,
            "interviewer": self.interviewer,
            "sections": [s["title"] for s in self.sections],
            "current_section": section["title"] if section else None,
            "section_idx": self.section_idx,
            "answered_count": self.answered_count,
            "total_questions": self.total_questions,
            "current_question": self.current_question,
            "is_followup": self.phase == "answering_followup",
        }
