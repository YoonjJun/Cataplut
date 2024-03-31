import axios from 'axios';

function onLogin() {
    const company_name = document.getElementById("company_name").value;
    const problem = document.getElementById('problem').value;
    const solution = document.getElementById('solution').value;
    print(company_name, problem, solution);
    
    axios({
        method: "POST",
        url: 'https://reqres.in/api/login',
        data: {
            "company_name": company_name,
            "problem": problem,
            "solution": solution
        }
    })
    .then((res) => {
        console.log(res);
    })
    .catch((error) => {
        console.error(error);
        // Handle the error appropriately
    });
}
