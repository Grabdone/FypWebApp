$(document).ready(function () {
    

    //Get the button:
    mybutton = document.getElementById("myBtn");

    // When the user scrolls down 20px from the top of the document, show the button
    window.onscroll = function() {scrollFunction()};

    function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        mybutton.style.display = "block";
    } else {
        mybutton.style.display = "none";
    }
    }

    // When the user clicks on the button, scroll to the top of the document
    $("#myBtn").click(function () {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
    });


    // Init
    $('#btn-predict').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#gender').text(data[1]);
                $('#age').text(data[2]);
                if(data[1]=="['male']" && data[2]=="['young']"){
                
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#maleYoung";                 //Go to the target element.
                    history.replaceState(null,null,url);

                }else if(data[1]=="['female']" && data[2]=="['young']"){
                
                   
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#femaleYoung";                 //Go to the target element.
                    history.replaceState(null,null,url);
                
                }else if(data[1]=="['male']" && data[2]=="['adult']"){
                
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#maleAdult";                 //Go to the target element.
                    history.replaceState(null,null,url);
                
                }else if(data[1]=="['female']" && data[2]=="['adult']"){
                
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#femaleAdult";                 //Go to the target element.
                    history.replaceState(null,null,url);
                
                }else if(data[1]=="['male']" && data[2]=="['senior']"){
                
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#maleSenior";                 //Go to the target element.
                    history.replaceState(null,null,url);

                }else if(data[1]=="['female']" && data[2]=="['senior']"){
                    
                    var url = location.href;               //Save down the URL without hash.
                    location.href = "#femaleSenior";                 //Go to the target element.
                    history.replaceState(null,null,url);
                }
                
                
                    
                console.log(data[1]);  
            },
        });
        
    });

});
