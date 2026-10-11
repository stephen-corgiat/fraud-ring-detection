/* Explainer lessons — reusable quiz widget.
   Markup contract:
     <div class="quiz">
       <p class="q">Question text</p>
       <button data-correct="false" data-fb="why this is wrong">Option A</button>
       <button data-correct="true"  data-fb="why this is right">Option B</button>
       <p class="feedback"></p>
     </div>
   Immediate feedback, retry allowed — a tight, automatic feedback loop. */

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".quiz").forEach((quiz) => {
    const fb = quiz.querySelector(".feedback");
    quiz.querySelectorAll("button[data-correct]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const correct = btn.dataset.correct === "true";
        quiz.querySelectorAll("button").forEach((b) => b.classList.remove("right", "wrong"));
        btn.classList.add(correct ? "right" : "wrong");
        if (fb) {
          fb.textContent = (correct ? "✓ " : "✗ ") + (btn.dataset.fb || "");
          fb.className = "feedback " + (correct ? "ok" : "no");
        }
      });
    });
  });
});
