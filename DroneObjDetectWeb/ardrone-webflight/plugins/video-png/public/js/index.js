(function(window, document) {

        var Video = function Video(cockpit) {
            console.log("Initializing video plugin.");
            
            // Add some UI elements
            $('#cockpit').append('<img id="video" src="" />');
            $('#cockpit').append('<img id="video1" src="" />');
            $('#cockpit').append('<img id="video2" src="" />');

            // Update image at 20fps
            var videoImg = $("#video");
            videoImg.attr("src", '/camera/' + new Date().getTime());

            setInterval(function() {
                videoImg.attr("src", '/camera/' + new Date().getTime());
            }, 100);
            
            var videoImg1 = $("#video1");
            videoImg1.attr("src", "/labeled/" + new Date().getTime());

            setInterval(function() {
                videoImg1.attr("src", "/labeled/" + new Date().getTime());
            }, 100);
            
            var videoImg2 = $("#video2");
            videoImg2.attr("src", "/depth/" + new Date().getTime());

            setInterval(function() {
                videoImg2.attr("src", "/depth/" + new Date().getTime());
            }, 100);
           
            
            
        };

        window.Cockpit.plugins.push(Video);
}(window, document));
