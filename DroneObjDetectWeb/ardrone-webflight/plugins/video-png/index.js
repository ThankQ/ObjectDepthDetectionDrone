fs = require('fs')

function video(name, deps) {
    
    var latestImage;
    
    var labeledImage;
    
    var depthImage;
    
    // Add a new route to fetch camera image
    deps.app.get('/camera/:id', function(req, res) {
      if (!latestImage) {
          res.writeHead(301, {"Location": "/plugin/" + name + "/images/nofeed.jpg"});
          res.end();
          return;
      }
      
      res.writeHead(200, {'Content-Type': 'image/png'});
      return res.end(latestImage, "binary");
    });
    
    deps.app.get('/camera_py', function(req, res) {
      res.writeHead(200, {'Content-Type': 'image/png'});
      return res.end(latestImage);
    });

    // Add a handler on images update
    deps.client.getPngStream()
      .on('error', console.log)
      .on('data', function(frame) { 
        latestImage = frame; 
    }); 
    
    // Labeled Image Stuff
    deps.app.post('/uploadLabeled', function(req, res) {
      console.log("START OF POST")
      
      res.writeHead(200, {'Content-Type': 'text/plain'});
      fs.readFile(req.body.url, function read(err, data){
        if(err){
          console.log(err)
          throw err;
        }
        console.log(data)
        labeledImage = data;           
      });
      res.end("");
    });
    
    deps.app.get('/labeled/:id', function(req, res) {
      if (!labeledImage) {
          res.writeHead(301, {"Location": "/plugin/" + name + "/images/nofeed.jpg"});
          res.end();
          return;
      }
      
      res.writeHead(200, {'Content-Type': 'image/png'});
      return res.end(labeledImage, "binary");
    });
    
     // Labeled Image Stuff
    deps.app.post('/uploadDepth', function(req, res) {
      console.log("START OF POST")
      
      res.writeHead(200, {'Content-Type': 'text/plain'});
      fs.readFile(req.body.url, function read(err, data){
        if(err){
          console.log(err)
          throw err;
        }
        console.log(data)
        depthImage = data;           
      });
      res.end("");
    });
    
    deps.app.get('/depth/:id', function(req, res) {
      if (!depthImage) {
          res.writeHead(301, {"Location": "/plugin/" + name + "/images/nofeed.jpg"});
          res.end();
          return;
      }
      
      res.writeHead(200, {'Content-Type': 'image/png'});
      return res.end(depthImage, "binary");
    });
};

module.exports = video;
