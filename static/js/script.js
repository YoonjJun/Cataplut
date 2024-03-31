document.getElementById("submit_btn").addEventListener("click", function(event) {
    event.preventDefault(); // Prevent the default form submission
 
    // Get the input values
    var companyName = document.getElementById("company_name").value;
    // Get other input values similarly
 
    // Create an object with the data
    var formData = {
       company_name: companyName,
       // Add other fields similarly
    };
 
    // Make an AJAX request
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/your-backend-endpoint-url", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
       if (xhr.readyState === 4 && xhr.status === 200) {
          // Request was successful
          console.log(xhr.responseText);
       }
       // Handle other status codes as needed
    };
    xhr.send(JSON.stringify(formData));
 });