// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.7.2/firebase-auth.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries




// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBNHa6mWVWWFOYZaTf9bW6dweCRJVS-FX0",
  authDomain: "health-chatbot-c0564.firebaseapp.com",
  projectId: "health-chatbot-c0564",
  storageBucket: "health-chatbot-c0564.appspot.com",
  messagingSenderId: "345885204105",
  appId: "1:345885204105:web:d4c244b2ad68c9a463b058"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Submit button
const submit = document.getElementById('submit');
submit.addEventListener("click", function(event){
  event.preventDefault();
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  const auth = getAuth();
  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      // Signed up 
      const user = userCredential.user;
      alert("Creating account...");
      // ...
    })
    .catch((error) => {
      const errorCode = error.code;
      const errorMessage = error.message;
      alert(errorMessage);
      // ..
    });
});