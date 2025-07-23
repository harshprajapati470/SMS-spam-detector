document.getElementById("spamForm").addEventListener("submit", async function(e) {
      e.preventDefault();
      const message = document.getElementById("sms").value;
      const response = await fetch( "http://127.0.0.1:5500/prediction" , {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
      });
      const data = await response.json();
      document.getElementById("result").textContent = `Result: ${data.prediction}`;
    });