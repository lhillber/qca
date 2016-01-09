QCAAdmin.controller('visualizer', ["$scope", "$rootScope",'$http',  function($scope,$rootScope,$http) {


    $scope.simcache = {}
    $scope.simlist = {} //index by unique color

    $scope.update = 0

    $rootScope.$watch(function() { return JSON.stringify($rootScope.selectedsims); },function() { 
        var tmplist = $scope.simlist
        $scope.simlist = {}



        for (var i = 0; i < $rootScope.selectedsims.length; i++) {
            var pk = $rootScope.selectedsims[i] 
            
            var color = $rootScope.colorforsim(pk)

            for (var key in tmplist) {
                if ($scope.simlist.hasOwnProperty(key) && tmplist[key].meta && tmplist[key].meta.pk == pk) {
                    $scope.simlist[color] = tmplist[key]
                    continue
                }
            }
            if ($scope.simlist.hasOwnProperty(color)) continue
            
            $scope.simlist[color] = {"loading":true}

            if ($scope.simcache.hasOwnProperty(pk)) {
                $scope.simlist[color] = $scope.simcache[pk]
            } else {
                console.log("requested: "+pk)
                $http.get('/simData/?pk='+pk,{}).then(function(response) {
                    var pk = JSON.parse(response.data.meta).pk
                    console.log("received: "+pk)
                    $scope.simcache[pk] = response.data
                    for (key in $scope.simcache[pk]) {
                        $scope.simcache[pk][key] = JSON.parse($scope.simcache[pk][key])
                    }
                    console.log("parsed: "+pk)

                    if ($rootScope.selectedsims.indexOf(pk) == -1) return
                    $scope.simlist[$rootScope.colorforsim(pk)] = response.data
                    $scope.update++
                },function(response) {
                    $scope.error = response.statusText
                })
            }
        }
            
        $scope.update++

    })

                 
    $scope.displays = {}
    $scope.toggle = function(idx) {
        if ($rootScope.selectedsims.length == 0) return
        if ($scope.displays[idx]) $scope.displays[idx] = false
        else $scope.displays[idx] = true
    } 

    

    $scope.domain = "time"
     
    /*
    $scope.displays = []
    $scope.updateDisplay = function(did) {
        $scope.displays[did] = []
        for (var it = 0; it < $scope.matrices[did].length; it++ ) {
            var display = []

            for (var i = 0; i < $scope.matrices[did][it].length; i++) {
                display.push({qbit: "", schmidt: "", maxmixed: false})

                var qubitvals = $scope.qbitcolor($scope.matrices[did][it][i])
                display[i].qbit = qubitvals[0]
                display[i].border = qubitvals[1]
                display[i].maxmixed = qubitvals[2]    

                if (i != $scope.matrices[did][it].length -1) {
                    var d = 255*(1- $scope.vonneumann[did][it][i] )

                    display[i].schmidt = "rgb("+Math.round(d)+","+Math.round(d)+","+Math.round(d)+")"
                }

            }

            $scope.displays[did].push(display)
        }
    }


    //density matrix -> color
    $scope.qbitcolor = function(density) {
       if (density[0][0].re + density[1][1].re - 1 > 1e-1) throw "matrix " +$scope.showDensity(density)+ " violates trace condition! ("+ JSON.stringify(density[0][0].re + density[1][1].re)   +"+" + JSON.stringify( density[0][0].im + density[1][1].im) + "i)"

       var ph = density[0][1].toPolar().phi/Math.PI

      var r = 0
      var g = 0
      var b = 0
       if (density[0][1].toPolar().r > 1e-3)  {
           var r = 255*(1 - Math.abs(ph)/(2/3))
           if (r < 0) r = 0
           
           ph = ph - 2/3
           if (ph < -1) ph = 2 +ph

           var g = 255*(1 - Math.abs(ph)/(2/3))
           if (g < 0) g = 0

           ph = ph - 2/3
           if (ph < -1) ph = 2 +ph

           var b = 255*(1 - Math.abs(ph)/(2/3))
           if (b < 0) b = 0
        }

       var len = (2*(density[0][0].re)-1)*(2*(density[0][0].re)-1)
       len += 4*density[0][1].re*density[0][1].re
       len += 4*density[0][1].im*density[0][1].im
       len = math.sqrt(len)

       var z= (2*(density[0][0].re) - 1)
       //if ( len > 1e-3) z = z
       
       if (z < 0) {
            r = (1+z)*r
            g = (1+z)*g
            b = (1+z)*b
       } else {
            r = r + (255 - r)*z
            g = g + (255 - g)*z
            b = b + (255 - b)*z
       }

        
        $scope.totParticles += (1-z)/2
        $scope.totVectors += (1-len)

        var border = Math.round(1 + (1-len)*11)
        
        var maxmixed = false
        if (len < 1e-3) maxmixed = true


        return ["rgb("+Math.round(r)+","+Math.round(g)+","+Math.round(b)+")",border,maxmixed]
    }
    */





}])


QCAAdmin.directive("plot", function ()
  {
    return {
        restrict: 'A',
        scope: {
            plot: '=',
            data: '=',
            title: '=',
            update: '=',
            /*select: '=',*/
            max: '=?',
        },
        template: "<canvas width='500px' height='300px'></canvas>",
        link: function($scope, element, attrs) {
           $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');
           

            if ($scope.max === undefined) $scope.max = 0.1


           $scope.$watch(function() { return $scope.update; },function() { 
                 
                 var list = {}
                 for (var color in $scope.simlist) {
                     list[color] = $scope.simlist[color][key]
                 }
                 return list
                    

                 var ctx = $scope.context
                 var canvas = $scope.canvas

                 //fill background 
                 ctx.fillStyle = "white";
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
                 
                 //draw title
                 ctx.fillStyle = "black";
                 ctx.font = "15px sans";
                 ctx.textAlign = "center";
                 ctx.fillText($scope.title, canvas.width/2, 20); 


                 var leftaxis = 30
                 //draw axes
                 ctx.beginPath(); 
                 ctx.lineWidth="2";
                 ctx.strokeStyle="black";
                 ctx.moveTo(leftaxis-5,canvas.height-15);
                 ctx.lineTo(canvas.width - 15,canvas.height-15);
                 ctx.stroke()

                 ctx.beginPath(); 
                 ctx.moveTo(leftaxis,canvas.height-10);
                 ctx.lineTo(leftaxis,30);
                 ctx.stroke()


                 //draw ticks
                 ctx.font = "10px sans";

                 var maxx = 5

                 for (color in $scope.plot) {
                    if ($scope.plot[color].length > maxx) maxx = $scope.plot[color].length
                 }

                 var divx = math.ceil(math.pow(10,math.log10(maxx)-1)/5)*5

                 var width = canvas.width - 15 - leftaxis
                 var divstep = math.pow(10,math.round(math.log10(maxx)-1))

                 var div = (width/maxx)*divstep
            
                 if (divstep == 0) divstep = 0.1
                 for (var x = 0; x <= maxx; x+=divstep) {
                     if (x == 0) continue
                     if (x%divx == 0) var l = 3
                     else var l = 2

                     ctx.beginPath(); 
                     ctx.moveTo(leftaxis + div*x/divstep, canvas.height-15-l);
                     ctx.lineTo(leftaxis + div*x/divstep, canvas.height-15+l);
                     ctx.stroke()
                    
                     if (x%divx == 0) {
                        ctx.fillText(x, leftaxis+div*x/divstep, canvas.height-3); 
                     }
                 }
                
                 return

                 var trunc = []
                 for (var i = 0; i < $scope.y.length; i++) {
                    if ($scope.y[i] !== undefined ) trunc.push($scope.y[i])
                 }

                 if (trunc.length >0 && (!Array.isArray(trunc[0]) ||  trunc[0].length >0   ) ) {
                     var maxy = math.ceil(math.max(trunc)*10)/10
                     if ($scope.max != 0)var maxy = math.max(math.ceil(maxy*10)/10,$scope.max)
                 } else if ($scope.max == 0) var maxy = 0.1
                 else var maxy = $scope.max
                
                     

                 var height = canvas.height - 45

                 var divstep = math.pow(10,math.round(math.log10(maxy)-1))
                
                 var count = 1
                 while (maxy/(divstep*count) > 15) count++
                 var divy = divstep*count
                 
                 var div = (height/maxy)*divstep
                   
                 var mult = 1
                 while (divy*mult < 1) mult *= 10
                 
                 if (divstep == 0) divstep = 0.1
                 for (var y = 0; y <= maxy; y+=divstep) {
                     if (y == 0) continue
                     if (y%divy == 0) var l = 3
                     else var l = 2

                     ctx.beginPath(); 
                     ctx.moveTo(leftaxis -l, canvas.height - 15 - div*y/divstep);
                     ctx.lineTo(leftaxis+l, canvas.height - 15 - div*y/divstep);
                     ctx.stroke()
                    
                     if ((math.round(y*mult,0))%(divy*mult) < 1e-3) {
                        ctx.fillText(math.round(y,2), leftaxis/2-3, canvas.height -12 - div*y/divstep); 
                     }
                 }


                 // draw points
                 var xstep = width/maxx
                 var ystep = height/maxy
              


                 var data = $scope.y

                 for (var idx = 0; idx < data.length; idx++) {
                     ctx.beginPath(); 
                     ctx.lineWidth="1";
                     ctx.strokeStyle= $scope.colorFor(idx)
                     ctx.fillStyle = $scope.colorFor(idx)


                     var rect = true
                     if (width/maxx < 6) rect = false

                     if (data[idx] === undefined) return

                     for (var i =0; i < data[idx].length; i++) {
                         if (data[idx][i] === undefined) continue
                         var xpos = leftaxis + xstep*$scope.x[idx][i]
                         var ypos = canvas.height - 15 - ystep*data[idx][i]

                         if (i == 0) {
                            ctx.moveTo(xpos,ypos);
                         } else {
                            ctx.lineTo(xpos,ypos);
                         }
                        
                        
                         if (rect) ctx.fillRect(xpos-2,ypos-2,4,4);
                         
                        ctx.fillStyle = "black"
                         if (i== $scope.select[1] && idx == $scope.select[0]) ctx.fillRect(xpos-3,ypos-3,6,6);
                        ctx.fillStyle = $scope.colorFor(idx)
                     
                     }

                    ctx.stroke()
                }
                 
           },true);
          
    }}
});




QCAAdmin.directive("network", function() {

            return {
                restrict: 'A',
                template: "<canvas width='520px' height='290px' style='border:1px solid black;'></canvas>",
                link: function($scope, $element, $attrs) {
                   
                    element[0].onclick = function() {
                        scope.$apply(function() {
                            if (scope.selectedRow < scope.networks[scope.selectedData].length -1) {
                                scope.selectedRow++
                                scope.binmax = 8
                            } else {
                                scope.selectedRow = -1
                                scope.selectedData = -1
                                scope.binmax = 8
                            }
                        })
                    }

                     scope.$watch(function() { 
                         return [scope.selectedData,scope.selectedRow]
                    },function() { 
                         if (!scope.networks[scope.selectedData]) return
                         parse = scope.networks[scope.selectedData][scope.selectedRow]
                        
                         if (!parse) return
                         

                         var canvas = element.find('canvas')[0];
                         var ctx = canvas.getContext('2d');



                         //fill background 
                         ctx.fillStyle = "white";
                         ctx.fillRect(0, 0, canvas.width, canvas.height); 
                        

                       
                         //draw title
                         ctx.fillStyle = "black";
                         ctx.font = "15px sans";
                         ctx.textAlign = "center";
                         ctx.fillText("Mutual Information Network for Iteration " + (scope.selectedRow), canvas.width/2, 20);


                         var dotPos =  function(i) {
                            return (10+widthper*i)
                         }



                             //draw dots
                             ctx.strokeStyle = "black"

                             
                             var height = canvas.height-10 //change for double-high
                             var widthper = (canvas.width-20)/(parse.length-1)
                             for (var i = 0; i < parse.length; i++) {
                             
                                ctx.fillStyle = scope.displays[scope.selectedData][scope.selectedRow][i].qbit;
                                 var x = math.floor(10+widthper*i-3) 
                                 var y = math.floor(height-3)
                                 ctx.fillRect(x,y,7,7);
                             }



                             //draw arcs
                            for (var i = 0; i < parse.length; i++) {
                                for (var j = i; j < parse.length; j++) {
                                    ctx.strokeStyle = "black"
                                                                                
                                    if (parse[i][j] < 1e-3) continue

                                    ctx.beginPath()
                                    var rad = (dotPos(j) - dotPos(i))/2
                                    ctx.lineWidth = 2*math.abs(parse[i][j])
                                    ctx.arc( dotPos(i) + rad ,height, rad, Math.PI, 2*Math.PI  )
                                    ctx.stroke()

                                }
                                   
                            }

                         
                    },true);
                }
            }})




QCAAdmin.directive("histogram", function() {

            return {
                restrict: 'A',
                template: "<canvas width='520px' height='290px' style='border:1px solid black;'></canvas>",
                link: function(scope, element, attrs) {

                    element[0].onclick = function() {
                        scope.$apply(function() {
                            if (scope.binmax > 1) scope.binmax--
                        })
                    }


                     scope.$watch(function() { 
                         return [scope.selectedData,scope.selectedRow,scope.binmax]
                    },function() { 
                         if (!scope.networks[scope.selectedData]) return
                         parse = scope.networks[scope.selectedData][scope.selectedRow]
                        
                         if (!parse) return
                        

                         var canvas = element.find('canvas')[0];
                         var ctx = canvas.getContext('2d');

                         //fill background 
                         ctx.fillStyle = "white";
                         ctx.fillRect(0, 0, canvas.width, canvas.height); 

                       
                         //draw title
                         ctx.fillStyle = "black";
                         ctx.font = "15px sans";
                         ctx.textAlign = "center";
                         ctx.fillText("Mutual Information Histogram for Iteration " + (scope.selectedRow), canvas.width/2, 20);

                         var leftaxis = 30
                         //draw axes
                         ctx.beginPath(); 
                         ctx.lineWidth="2";
                         ctx.strokeStyle="black";
                         ctx.moveTo(leftaxis-5,canvas.height-15);
                         ctx.lineTo(canvas.width - 15,canvas.height-15);
                         ctx.stroke()

                         ctx.beginPath(); 
                         ctx.moveTo(leftaxis,canvas.height-10);
                         ctx.lineTo(leftaxis,30);
                         ctx.stroke()

                         var perbin = 0
                         var occupiedBins = 0

                         var bestperbin = 1
                         var bestOccupiedBins = 1

                         while (perbin < scope.binmax) {
                             perbin++
                        
                             // compute histogram
                             var hist = []
                             for (var i = 0; i*0.1/perbin < 1; i++) hist.push(0)

                             for (var i = 0; i < parse.length; i++) {
                                    for (var j = i; j < parse.length; j++) {
                                        for (var k = 0; k*0.1/perbin < 1; k++) {
                                            if (k*0.1/perbin <= parse[i][j] &&  parse[i][j] < (k+1)*0.1/perbin) {
                                                hist[k]++
                                                break
                                            }
                                        }
                                           if (parse[i][j] == 1 && 1 == (k+1)*0.1/perbin ) {
                                                hist[k]++
                                                break
                                            }

                                    }
                                       
                                }
                        
                            occupiedBins = 0
                            for (var i = 0; i*0.1/perbin < 1; i++) {
                                if (hist[i] != 0) occupiedBins++;
                            }
                            
                            if (occupiedBins > bestOccupiedBins) {
                                bestperbin = perbin
                                bestOccupiedBins = occupiedBins
                            } else {
                            
                            }
                         }

                         perbin = bestperbin

                        var hist = []
                             for (var i = 0; i*0.1/perbin < 1; i++) hist.push(0)

                             for (var i = 0; i < parse.length; i++) {
                                    for (var j = i; j < parse.length; j++) {
                                        for (var k = 0; k*0.1/perbin < 1; k++) {
                                            if (k*0.1/perbin <= parse[i][j] &&  parse[i][j] < (k+1)*0.1/perbin) {
                                                hist[k]++
                                                break
                                            }
                                            if (parse[i][j] == 1 && 1 == (k+1)*0.1/perbin ) {
                                                hist[k]++
                                                break
                                            }
                                        }

                                    }
                                       
                                }


                         //draw ticks
                         ctx.font = "10px sans";

                         var maxx = 1


                         var width = canvas.width - 15 - leftaxis
                         var divstep = 0.1

                         var div = (width/maxx)*divstep
                    
                         if (divstep == 0) divstep = 0.1
                         for (var x = 0; x <= maxx; x+=divstep) {
                             if (x == 0) continue
                             else var l = 2

                             ctx.beginPath(); 
                             ctx.moveTo(leftaxis + div*x/divstep, canvas.height-15-l);
                             ctx.lineTo(leftaxis + div*x/divstep, canvas.height-15+l);
                             ctx.stroke()
                            
                             ctx.fillText(math.round(x,1), leftaxis+div*x/divstep, canvas.height-3); 
                         }
                      


                         var first = hist.shift()
                         var maxy = math.max(hist)
                         if (maxy == 0) maxy = 1
                         hist.unshift(first)
                        
                             

                         var height = canvas.height - 45

                         var divstep = math.pow(10,math.round(math.log10(maxy)-1))
                        
                         var count = 1
                         while (maxy/(divstep*count) > 15) count++
                         var divy = divstep*count
                         
                         var div = (height/maxy)*divstep
                           
                         var mult = 1
                         while (divy*mult < 1) mult *= 10
                         
                         if (divstep == 0) divstep = 0.1
                         for (var y = 0; y <= maxy; y+=divstep) {
                             if (y == 0) continue
                             if (y%divy == 0) var l = 3
                             else var l = 2

                             ctx.beginPath(); 
                             ctx.moveTo(leftaxis -l, canvas.height - 15 - div*y/divstep);
                             ctx.lineTo(leftaxis+l, canvas.height - 15 - div*y/divstep);
                             ctx.stroke()
                            
                             if ((math.round(y*mult,0))%(divy*mult) < 1e-3) {
                                ctx.fillText(math.round(y,2), leftaxis/2-3, canvas.height -12 - div*y/divstep); 
                             }
                         }


                            
                         ctx.fillStyle = scope.colorFor(scope.selectedData)
                         ctx.strokeStyle = "black"
                         ctx.lineWidth = 1

                         var xCoord = function(i) {
                            return leftaxis + (width/maxx)*(i*0.1/perbin)
                         }

                         var yCoord = function(i) {
                            return canvas.height-15 - (height/maxy)*i
                         }


                         var barwidth = (width/maxx)*0.1/perbin
                         for (var i = 0; i*0.1/perbin < 1; i++) {
                            if (perbin == 1) {
                                ctx.fillRect(xCoord(i)+barwidth/4,yCoord(hist[i]), barwidth/2 ,  (height/maxy)*hist[i] ) 
                                ctx.strokeRect(xCoord(i)+barwidth/4,yCoord(hist[i]),barwidth/2 ,  (height/maxy)*hist[i] ) 
                            } else {
                                ctx.fillRect(xCoord(i),yCoord(hist[i]), barwidth ,  (height/maxy)*hist[i] ) 
                                ctx.strokeRect(xCoord(i),yCoord(hist[i]),barwidth ,  (height/maxy)*hist[i] ) 
                            }
                            
                         }


                         
                    },true);
                }
            }})


