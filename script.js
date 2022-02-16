// Loads the trained model from my github to this HTML file as "model".
async function loadModel() {
    model = undefined
    model = await tf.loadLayersModel("https://raw.githubusercontent.com/KungFuTurtlePooh/model/main/model.json");
    console.log("model loaded");
}
loadModel();

// predict function which gets the image from the "previewImage" ID and sets it as a tensor. The new tensorImage is then
// passed into the model using model.predict. After, we get a list of 4 weights, the largest one being the one the AI believes
// is most likely correct. It uses a for loop to loop these weights and store the index of the highest weight.  Finally,
// it gets a character name from te characters list and sets the "results" ID to whatever character it predicts.
async function predict() {
    let image = document.getElementById("previewImage")
    let tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([180, 180]).toFloat().expandDims();

    prediction = await model.predict(tensorImg).data();

    var characters = ["Vladilena Milizé", "Miku Nakano", "Shinoa Hiiragi", "Tsukasa Tsukuyomi"];
    var max = prediction[0];
    var index = 0;
    for(var i = 0; i < 4; i++)
    {
        if (prediction[i] > max)
        {
            max = prediction[i];
            index = i;
        }
    }
    predictedCharacter = characters[index];


    character = predictedCharacter;
                document.getElementById("results").innerHTML = "A.I. Prediction: ";
                t = 0;
                typeWriterEffect();
}

// There are 3 global variables defined here: t, speed, and character.  The function right below creates a typewriter effect
// which loops through the entire name of the predicted character name and adds each character from the name one by one.
var t = 0;
var speed = 45;
var character;
function typeWriterEffect() {
    if (t < character.length){
        document.getElementById("results").innerHTML += character.charAt(t);
        t++;
        setTimeout(typeWriterEffect, speed);
    }
}


// Event listener which sees if the img input is changed. If it is, then
// it creates an instance of FileReader() and uses that to save an inputted
// image to the local storage of the website. Then it sets the src of the 
// preview image to whatever image is saved on the local storage
document.querySelector("#img").addEventListener("change", function () {
    const fileReader = new FileReader();

    fileReader.addEventListener("load", () => {
        localStorage.setItem("inputImage", fileReader.result);
        const dataImage = localStorage.getItem("inputImage");
        document.getElementById("previewImage").src=dataImage;
    });

    fileReader.readAsDataURL(this.files[0]);
})
