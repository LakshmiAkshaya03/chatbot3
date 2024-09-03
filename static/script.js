const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

let userMessage = null; // Variable to store user's message
const API_URL = "http://127.0.0.1:5000/api/chatbot"; // URL of your Flask backend
const inputInitHeight = chatInput.scrollHeight;

// Function to speak out the chatbot responses without *
function speakResponse(responseText) {

    const cleanResponseText = responseText.replace(/\*/g, '');
    
    var speech = new SpeechSynthesisUtterance();
    speech.text = cleanResponseText;
    speech.volume = 1;
    speech.rate = 1;
    speech.pitch = 1;

    // Speak the response
    window.speechSynthesis.speak(speech);
}

const createChatLi = (message, className) => {

    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi; // return chat <li> element
}

const generateResponse = (message) => {
    
    const incomingChatLi = createChatLi("Predicting...", "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);


    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ symptoms: message })
    })
    .then(response => response.json())
    .then(data => {
        incomingChatLi.querySelector("p").textContent = data.message;
        
        speakResponse(data.message);
    })
    .catch(error => {
        incomingChatLi.querySelector("p").textContent = "Oops! Something went wrong. Please try again.";
        
        speakResponse("Oops! Something went wrong. Please try again.");
    })
    .finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
}

const handleChat = () => {
    userMessage = chatInput.value.trim(); 
    if(!userMessage) return;


    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;


    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    
    setTimeout(() => generateResponse(userMessage), 600);
}

chatInput.addEventListener("input", () => {

    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {

    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));