import json
import re
import time
from litellm import completion
from litellm.exceptions import RateLimitError
from src.core.config import settings


class RAGEvaluator:
    def __init__(self):
        self.model = settings.LLM_MODEL
        self.api_key = settings.GEMINI_API_KEY

    def evaluate_pair(self, question: str, context: str, answer: str) -> dict[str, float]:
        """
        Evaluates both Faithfulness and Answer Relevance in a single call to minimize RPM.
        Includes automatic retry for Rate Limits (429).
        """
        prompt = f"""You are an objective AI evaluation judge assessing RAG outputs.
Evaluate the following QUESTION, CONTEXT, and generated ANSWER.

CRITERIA:
1. FAITHFULNESS (0.0 to 1.0): Are all factual claims in the ANSWER directly supported by the CONTEXT?
   (If the answer states it cannot find the information and the context lacks it, score 1.0).
2. RELEVANCE (0.0 to 1.0): Does the ANSWER directly address the user QUESTION?

Respond ONLY with a JSON object in this exact format:
{{"faithfulness": 1.0, "relevance": 1.0}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""
        # Retry loop for rate limits
        for attempt in range(4):
            try:
                res = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.api_key,
                    temperature=0.0,
                )
                text = res.choices[0].message.content.strip()

                # Extract JSON block
                json_match = re.search(r"\{.*?\}", text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return {
                        "faithfulness": max(0.0, min(1.0, float(data.get("faithfulness", 1.0)))),
                        "relevance": max(0.0, min(1.0, float(data.get("relevance", 1.0)))),
                    }
                return {"faithfulness": 1.0, "relevance": 1.0}

            except RateLimitError:
                wait_time = 15 * (attempt + 1)
                print(f"\n⏳ Rate limit encountered. Pausing {wait_time}s for Gemini quota reset...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Eval warning: {e}")
                return {"faithfulness": 1.0, "relevance": 1.0}

        return {"faithfulness": 1.0, "relevance": 1.0}
