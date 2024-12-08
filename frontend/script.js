const apiUrl = "https://wdjy6n2uh8ffq7-8000.proxy.runpod.net/generate?prompt=";
const imageForm = document.getElementById('imageForm');
const promptInput = document.getElementById('promptInput');
const imageContainer = document.getElementById('imageContainer');

imageForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const prompt = promptInput.value.trim();
    console.log("Prompt submitted:", prompt); // Debug: Print the submitted prompt
    if (!prompt) {
        console.warn("No prompt entered"); // Debug: Warn if prompt is empty
        return;
    }

    // Clear any previous image or error
    imageContainer.innerHTML = '';
    console.log("Cleared previous image or messages"); // Debug: Confirm UI reset

    // Show loading text
    const loadingText = document.createElement('p');
    loadingText.textContent = 'Generating image...';
    imageContainer.appendChild(loadingText);
    console.log("Displayed loading message"); // Debug: Confirm loading text displayed

    try {
        console.log("Sending request to API:", apiUrl + encodeURIComponent(prompt)); // Debug: Log the full API URL
        const response = await fetch(`${apiUrl}${encodeURIComponent(prompt)}`);
        console.log("API response received:", response); // Debug: Log raw response object

        // Check for HTTP errors
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Parsed API response:", data); // Debug: Log parsed JSON data

        if (data.message === 'Image generated successfully') {
            const img = document.createElement('img');
            img.src = `https://wdjy6n2uh8ffq7-8000.proxy.runpod.net${data.image_url}`;
            img.alt = "Generated image based on the prompt";
            imageContainer.innerHTML = ''; // Clear loading text
            imageContainer.appendChild(img);
            console.log("Image displayed successfully:", img.src); // Debug: Confirm image display
        } else {
            throw new Error(data.message || 'Image generation failed');
        }
    } catch (error) {
        console.error("Error occurred:", error); // Debug: Log the error
        imageContainer.innerHTML = `<p id="error">Error: ${error.message}</p>`;
    }
});
