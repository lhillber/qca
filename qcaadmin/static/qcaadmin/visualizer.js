QCAAdmin.controller('visualizer', ["$scope", "$rootScope",'$http',  function($scope,$rootScope,$http) {


    $scope.simcache = {}
    $scope.simlist = {} //index by unique color

    $scope.update = 0

    $scope.cutsunavailable = false

    $rootScope.$watch(function() { return JSON.stringify($rootScope.selectedsims); },function() { 
        var tmplist = $scope.simlist
        $scope.simlist = {}

        $scope.bubsettings.network = 0

        for (var i = 0; i < $rootScope.selectedsims.length; i++) {
            var pk = $rootScope.selectedsims[i] 
            
            var color = $rootScope.colorforsim(pk)
            $scope.maxlength = 0

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
                   for (color in $scope.simlist) {
                        if ($scope.simlist[color].loading) continue
                        var len = $scope.simlist[color]["one_site"][0].length 
                        if (len > $scope.maxlength )$scope.maxlength = len
                    }
            } else {
                console.log("requested: "+pk)
                $http.get(window.prefix+'/simData/?pk='+pk,{}).then(function(response) {
                    var pk = JSON.parse(response.data.meta).pk
                    console.log("received: "+pk)
                    $scope.simcache[pk] = response.data

                    $scope.cutsunavailable = true
                    for (key in $scope.simcache[pk]) {
                        if (key == 'sc') $scope.cutsunavailable = false
                        $scope.simcache[pk][key] = JSON.parse($scope.simcache[pk][key])
                    }


                    /*

                    var sc = []
                    
                    for (key in $scope.simcache[pk]) {
                        if (!key.includes("cut")) continue
                        var idx = parseInt(key.substr(3,key.length))
                        sc[idx] = $scope.simcache[pk][key]     
                        delete $scope.simcache[pk][key]
                    }
                    $scope.simcache[pk]['sc'] = sc
                    */
                    
                    console.log("parsed: "+pk)

                    if ($rootScope.selectedsims.indexOf(pk) == -1) return

                    $scope.simlist[$rootScope.colorforsim(pk)] = response.data
                    
                    for (color in $scope.simlist) {
                        if ($scope.simlist[color].loading) continue
                        var len = $scope.simlist[color]["one_site"][0].length 
                        if (len > $scope.maxlength )$scope.maxlength = len
                    }

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
    $scope.avgmode = "time"
    
    $scope.bubsettings = {
        "sites": "color",//'color', 'X', 'Y', 'Z', or 'none'
        "siteent": true,
        "cutent": true,
        "network":0,
    }

    $scope.maxlength = 0
    $scope.chNetworkPos = function(dir) {
        if (dir == '+' && $scope.bubsettings.network < $scope.maxlength-1) $scope.bubsettings.network++
        if (dir == '-' && $scope.bubsettings.network > 0) $scope.bubsettings.network--
        if (Number.isInteger(dir)) $scope.bubsettings.network = dir
    }



    $scope.netmode = 'm'
    $scope.setnet = function(net) {
        $scope.netmode = net
        $scope.update++
    }
    $scope.nethist = 'arcs'

}])


QCAAdmin.directive("plot", function ()
  {
    return {
        restrict: 'A',
        scope: {
            plot: '=',
            fplot: '=',
            simlist: '=',
            title: '=',
            update: '=',
            domain: '=',
            std:'=?',
            freqs:'=?',
            /*select: '=',*/
            max: '=?',
            xaxis: '=?',
        },
        template: "<canvas width='500px' height='300px'></canvas>",
        link: function($scope, element, attrs) {
           $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');
           

            if ($scope.max === undefined) $scope.max = 0.01


            $scope.redraw = function() { 
                 var ctx = $scope.context
                 var canvas = $scope.canvas

                 //fill background 
                 ctx.fillStyle = "white";
                 ctx.fillRect(0, 0, canvas.width, canvas.height);

                 var unavailable = true

                 var data = {}
                 var freqs = false
                 for (var color in $scope.simlist) {
                     
                     if ($scope.simlist[color].loading) continue
                     if ($scope.domain == 'time') {
                          var out =  $scope.simlist[color]
                          for (var i = 0; i < $scope.plot.length; i++) {
                              if (out === undefined) continue
                              out = out[$scope.plot[i]]
                          }
                          if (out === undefined) continue
                          unavailable = false
                         data[color] = out
                     } else {
                         var out =  $scope.simlist[color]
                         for (var i = 0; i < $scope.plot.length; i++) {
                            if (out === undefined) continue
                            out = out[$scope.fplot[i]]
                         }
                         if (out === undefined) continue
                         unavailable = false
                         data[color] = out

                         if ($scope.freqs !== undefined) {
                             freqs = $scope.simlist[color]
                             for (var i = 0; i < $scope.freqs.length; i++) {
                                if (freqs === undefined) continue
                                freqs = freqs[$scope.freqs[i]]
                             }
                             if (freqs === undefined) continue
                         }
                         else freqs = $scope.simlist[color]['freqs']
                    }
 
                 }

            

                 if (unavailable) {
                     ctx.fillStyle = "black";
                     ctx.font = "15px sans";
                     ctx.textAlign = "center";
                     ctx.fillText($scope.title + ' unavailable', canvas.width/2, canvas.height/2); 
                     return
                 
                 } 
                 
                 //draw title
                 ctx.fillStyle = "black";
                 ctx.font = "15px sans";
                 ctx.textAlign = "center";
                 if ($scope.domain == 'time') ctx.fillText($scope.title, canvas.width/2, 20); 
                 else ctx.fillText($scope.title+" Fourier Transform", canvas.width/2, 20); 


                 var leftaxis = 45
                 var botaxis = 32
                 //draw axes
                 ctx.beginPath(); 
                 ctx.lineWidth="2";
                 ctx.strokeStyle="black";
                 ctx.moveTo(leftaxis-5,canvas.height-botaxis);
                 ctx.lineTo(canvas.width - 15,canvas.height-botaxis);
                 ctx.stroke()

                 ctx.beginPath(); 
                 ctx.moveTo(leftaxis,canvas.height-botaxis+5);
                 ctx.lineTo(leftaxis,30);
                 ctx.stroke()

                 //draw ticks
                 ctx.font = "10px sans";

                 var maxx = 5

                 var width = canvas.width - 15 - leftaxis
                 if ($scope.domain == 'time') {
                     for (color in data) {
                        if (data[color].length > maxx) maxx = data[color].length
                     }  
                    var divx = math.ceil(math.pow(10,math.log10(maxx)-1)/5)*5
                    var divstep = math.pow(10,math.round(math.log10(maxx)-1))
                 } else {
                     maxx = freqs[freqs.length-1] 
                     var divx = freqs[1]
                     var divstep = freqs[1]
                 }


                 var div = (width/maxx)*divstep
                 if (divstep == 0) divstep = 0.1
    
                 if ($scope.domain == 'time') {
                     for (var x = 0; x <= maxx; x+=divstep) {
                         if (x == 0) continue
                         if (x%divx == 0) var l = 3
                         else var l = 2

                         ctx.beginPath(); 
                         ctx.moveTo(leftaxis + div*x/divstep, canvas.height-botaxis-l);
                         ctx.lineTo(leftaxis + div*x/divstep, canvas.height-botaxis+l);
                         ctx.stroke()
                        
                         //if (x%divx == 0) {
                            ctx.fillText(x, leftaxis+div*x/divstep, canvas.height-botaxis+13); 
                         //}
                     }
                 } else {
                     for (var i = 0; i < freqs.length; i++) {
                         if (i == 0) continue
                         var l = 2
                         var x = freqs[i]

                         ctx.beginPath(); 
                         ctx.moveTo(leftaxis + div*x/divstep, canvas.height-botaxis-l);
                         ctx.lineTo(leftaxis + div*x/divstep, canvas.height-botaxis+l);
                         ctx.stroke()
                         if (i%4 == (freqs.length-1)%4) ctx.fillText(math.round(x,2), leftaxis+div*x/divstep, canvas.height-botaxis+13); 
                     }
                 
                 }
                
                
                 //y ticks
                 var maxy = $scope.max

                 for (color in data) {
                    var max = math.max(data[color])
                    if (max > maxy) maxy = max
                 }

                 var height = canvas.height - 30 - botaxis

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
                     ctx.moveTo(leftaxis -l, canvas.height - botaxis - div*y/divstep);
                     ctx.lineTo(leftaxis+l, canvas.height - botaxis - div*y/divstep);
                     ctx.stroke()
                    
                     if ((math.round(y*mult,0))%(divy*mult) < 1e-3) {
                        ctx.fillText(math.round(y,2), leftaxis-15, canvas.height -botaxis+3 - div*y/divstep); 
                     }
                 }


                 // draw axis labels
                 ctx.font = "13px sans";
                 
                 var keyword = "Iterations"
                 if ($scope.xaxis !== undefined) {
                    if ($scope.xaxis == "time") keyword = "Iterations"
                    if ($scope.xaxis == "space") keyword = "Distance"
                 } 

                 var xaxis = "Frequency (Inverse "+keyword+")"
                 if ($scope.domain == 'time') xaxis = keyword

                 ctx.fillText(xaxis,leftaxis-5 + (canvas.width-10-leftaxis)/2   , canvas.height-5); 
                
                 ctx.translate(15,canvas.height/2)
                 ctx.rotate(-Math.PI/2)
                 ctx.fillText($scope.title,0,0); 
                 ctx.rotate(Math.PI/2)
                 ctx.translate(-15,-canvas.height/2)



                 // draw points
                 var xstep = width/maxx
                 var ystep = height/maxy

                 std = {}
                 hasstd = false
                 if ($scope.std != undefined && $scope.domain == 'time') {
                    for (color in data) {
                        stdc = $scope.simlist[color]
                        for (var i = 0; i < $scope.std.length; i++) {
                            if (stdc === undefined) continue
                            stdc = stdc[$scope.std[i]]
                        }
                        std[color] = stdc
                        hasstd = true
                    } 
                 }

                 for (color in data) {
                     ctx.beginPath(); 
                     ctx.lineWidth="1";
                     ctx.strokeStyle= color
                     ctx.fillStyle = color


                     var rect = true
                     if (width/maxx < 6) rect = false


                     for (var i =0; i < data[color].length; i++) {
                         if (data[color][i] === null) continue

                         if ($scope.domain == 'time') var xpos = leftaxis + xstep*(i)
                         else var xpos = leftaxis + xstep*(freqs[i])

                         var ypos = canvas.height - botaxis  - ystep*data[color][i]

                         if (i == 0) {
                            ctx.moveTo(xpos,ypos);
                         } else {
                            ctx.lineTo(xpos,ypos);
                         }
                       
                         if (std[color] !== undefined) {
                            ctx.fillRect(xpos-2,ypos-ystep*std[color][i],4,ystep*std[color][i]*2);
                         } else {
                           if (rect && !hasstd) ctx.fillRect(xpos-2,ypos-2,4,4);
                         }
                         
                        //ctx.fillStyle = "black"
                        // if (i== $scope.select[1] && idx == $scope.select[0]) ctx.fillRect(xpos-3,ypos-3,6,6);
                        //ctx.fillStyle = $scope.colorFor(idx)
                     
                     }

                    ctx.stroke()
                }
                 
           }
            $scope.$watch(function() { return $scope.update; },$scope.redraw,true);
            $scope.$watch(function() { return $scope.domain; },$scope.redraw,true);
            $scope.$watch(function() { return $scope.plot; },$scope.redraw,true);
          
    }}
});

QCAAdmin.directive("bubbles", function ($rootScope)
  {
    return {
        restrict: 'A',
        scope: {
            bubbles: '=',
            color: '=',
            settings: '=',
            update: '=',
            data: '=',
            network: '=',
            /*select: '=',*/
        },
        template: "<canvas width='250px' height='400px'></canvas>",
        link: function($scope, element, xattrs) {
           $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');
          



            $scope.scrollpos = 0
           element[0].addEventListener("wheel", function(e) {
                 e.stopPropagation()
                e.preventDefault()
                e.returnValue = false
                var speed = 2
                if (e.deltaY > 0) $scope.scrollpos+=speed
                if (e.deltaY < 0) {
                    if ($scope.scrollpos == 0) return
                    $scope.scrollpos-=speed
                }
   
                
                $rootScope.$broadcast('scrollpos',$scope.scrollpos)
                return false
           })


           $scope.redraw = function()  { 
                 if ($scope.bubbles.loading) return
                 
                 var leftaxis = 38
                 var rightmargin = 14
                 var botaxis = 30
                 var tabheight = 15 // part of topaxis
                 var topaxis = 30 + tabheight

                 var ctx = $scope.context
                 var canvas = $scope.canvas

                 //fill background 
                 ctx.fillStyle = "white";
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
                 
                 //draw title
                 ctx.fillStyle = "black";
                 ctx.font = "15px sans";
                 ctx.textAlign = "center";
                 ctx.fillText("Time Series", canvas.width/2, tabheight+20); 

                 //draw tab
                 ctx.fillStyle = $scope.color
                 ctx.fillRect(0,0,canvas.width,tabheight)
                 ctx.fillStyle = "black";

                 // draw axis labels
                 ctx.font = "13px sans";
                 ctx.fillText("Position",leftaxis-5 + (canvas.width-10-leftaxis)/2   , canvas.height-5); 
                
                 ctx.translate(15,canvas.height/2)
                 ctx.rotate(-Math.PI/2)
                 ctx.fillText("Iteration",0,0); 
                 ctx.rotate(Math.PI/2)
                 ctx.translate(-15,-canvas.height/2)

                 //determine sizes
                 var width = canvas.width - leftaxis - rightmargin
                 var height = canvas.height - topaxis - botaxis

                 var statelength = $scope.bubbles.meta.length
                 var boxdim = width/statelength
                 var numheight = (height - (height%boxdim))/boxdim


                 var boxx = function(j) { return leftaxis + j*boxdim  } 
                 var boxy = function(i) { return topaxis + (i-$scope.scrollpos)*boxdim } 

            

                 //draw numbers
                 ctx.fillStyle = "black"
                 ctx.font = math.round(boxdim*0.7) + "px sans";
                 ctx.textAlign = "right";
                 for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                    ctx.fillText(i,leftaxis - boxdim/4,boxy(i)+3*boxdim/4); 
                 }

                 ctx.textAlign = "center";
                 for (var j = 0; j < statelength  ; j++) {
                    ctx.fillText(j,boxx(j)+boxdim/2,canvas.height - botaxis + 3*boxdim/4); 
                 }


                    /*
                    $scope.bubsettings = {
                            "sites": "color"//'color', 'X', 'Y', 'Z', or 'none'
                            "siteent": true//site entropy
                            "cutent": true//site entropy
                        
                        }
                    */


                 //draw content
                 $scope.strokeStyle = "black"
                
                 if ($scope.settings.cutent && $scope.bubbles["sc"] !== undefined ) {
                     //var maxcut = 0
                     for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                        for (var j = 0; j+1 < statelength  ; j++) {
                            if ($scope.bubbles["sc"][i] === undefined) continue
                            if (Math.abs($scope.bubbles["sc"][i][j]*4) < 1e-3) continue
                            //if ($scope.bubbles["sc"][i][j] > maxcut) maxcut =$scope.bubbles["sc"][i][j]
                            ctx.lineWidth = Math.abs($scope.bubbles["sc"][i][j]*4)
                            ctx.beginPath();
                            ctx.moveTo(boxx(j)+boxdim/2,boxy(i)+boxdim/2)
                            ctx.lineTo(boxx(j+1)+boxdim/2,boxy(i)+boxdim/2)
                            ctx.stroke();
                              
                        }
                     }
                     //console.log("maxcut",maxcut)
                 }

                
                 var radius = boxdim/2
                 if (!$scope.settings.cutent) radius = boxdim/Math.sqrt(2)


                 for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                    for (var j = 0; j < statelength  ; j++) {
                        if ($scope.bubbles["one_site"][i] === undefined) continue

                        var density = $scope.bubbles["one_site"][i][j]
                        if ($scope.settings.sites == 'color') { 
                            var bubble = $scope.qbitcolor(density)
                            if (!bubble[1]) ctx.fillStyle= bubble[0]
                            else {
                                var grd=ctx.createRadialGradient(boxx(j)+boxdim/2,boxy(i)+boxdim/2,0,boxx(j)+boxdim/2,boxy(i)+boxdim/2,radius);
                                grd.addColorStop(0,"white");
                                grd.addColorStop(1,"black");
                                ctx.fillStyle = grd
                            }
                        } else if ($scope.settings.sites != 'none') {
                            setting = $scope.settings.sites 
                            if (setting == 'X') {
                                var value = 255-Math.round(255* (density[0][1].re + density[1][0].re) )
                                ctx.fillStyle = "rgb("+value+","+value+","+value+")"
                            }
                            if (setting == 'Y') {
                                var value = 255-Math.round(255* (density[1][0].im - density[0][1].im) )
                                ctx.fillStyle = "rgb("+value+","+value+","+value+")"
                            }
                            if (setting == 'Z') {
                                var value = Math.round(255* (density[0][0].re - density[1][1].re) )
                                ctx.fillStyle = "rgb("+value+","+value+","+value+")"
                            }
                            if (setting == 'Network') {
                                var net = $scope.data[$scope.color][$scope.network][i][$scope.settings.network]
                                var max = 0.2
                                for (var k = 0; k < statelength  ; k++) if (net[k] > max) max = net[k]

                                var value = 255-Math.round(255* (  net[j]/max  ))
                                ctx.fillStyle = "rgb("+value+","+value+","+value+")"
                                //ctx.fillStyle = "#F0F"
                            }
                        }

                        if ($scope.settings.sites != 'none' ) { 
                            if ($scope.settings.cutent)  {
                                ctx.beginPath();
                                ctx.arc(boxx(j)+boxdim/2,boxy(i)+boxdim/2,4*boxdim/10,0,2*Math.PI);
                                ctx.fill();
                            } else ctx.fillRect(boxx(j),boxy(i),boxdim,boxdim)
                        }

                        if (  !(bubble && bubble[1]) && $scope.settings.siteent ) {
                            if ($scope.settings.sites != 'none') {
                                var w = $scope.bubbles["s"][i][j]*2.5
                                if (w < 1e-3) continue
                                ctx.lineWidth = w 
                                if ($scope.settings.cutent)  {
                                    ctx.beginPath();
                                    ctx.arc(boxx(j)+boxdim/2,boxy(i)+boxdim/2,4*boxdim/10,0,2*Math.PI);
                                    ctx.stroke();
                                } else ctx.strokeRect(boxx(j)+w/2,boxy(i)+w/2,boxdim-w/2,boxdim-w/2)
                            } else {
                                var w = $scope.bubbles["s"][i][j]*4*boxdim/10
                                if (w < 1e-3) continue
                                ctx.fillStyle = "black"
                                if ($scope.settings.cutent)  {
                                    ctx.beginPath();
                                    ctx.arc(boxx(j)+boxdim/2,boxy(i)+boxdim/2,w,0,2*Math.PI);
                                    ctx.fill();
                                } else ctx.fillRect(boxx(j)+boxdim/2-w,boxy(i)+boxdim/2-w,w*2,w*2)
                            
                            }
                        }


                 
                    }
                 }

                 if ($scope.settings.sites == 'none' && !$scope.settings.siteent && !$scope.settings.cutent) {
                    $scope.fillStyle = "black"
                     ctx.font = "15px sans";
                    ctx.fillText("Nothing Selected",leftaxis + width/2  , topaxis + height/2); 
                 
                 }

                 
           }
           
           $scope.$watch(function() { return $scope.update; },$scope.redraw,true);
           $scope.$watch(function() { return $scope.settings; },$scope.redraw,true);
           $scope.$on('scrollpos',function(e,pos) {
               $scope.scrollpos = pos
               $scope.redraw()
           }, true)

            $scope.qbitcolor = function(density) {
               if (density[0][0].re + density[1][1].re - 1 > 1e-1) {
                   throw "matrix " +$scope.showDensity(density)+ " violates trace condition! ("+ JSON.stringify(density[0][0].re + density[1][1].re)   +"+" + JSON.stringify( density[0][0].im + density[1][1].im) + "i)"
               }

               var ph = math.complex(density[0][1]).toPolar().phi/Math.PI

               var r = 0
               var g = 0
               var b = 0
               if (math.complex(density[0][1]).toPolar().r > 1e-3)  {
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


                
                var maxmixed = false
                if (len < 1e-2) maxmixed = true


                return ["rgb("+Math.round(r)+","+Math.round(g)+","+Math.round(b)+")",maxmixed]
            }

          
    }}
});




QCAAdmin.directive("networks", function ($rootScope)
  {
    return {
        restrict: 'A',
        scope: {
            networks: '=',
            color: '=',
            update: '=',
            key: '=',
            mode: '=',
            /*select: '=',*/
        },
        template: "<canvas width='250px' height='400px'></canvas>",
        link: function($scope, element, xattrs) {
           $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');
          



            $scope.scrollpos = 0
           element[0].addEventListener("wheel", function(e) {
                e.stopPropagation()
                e.preventDefault()
                e.returnValue = false
                var speed = 2
                if (e.deltaY > 0) $scope.scrollpos+=speed
                if (e.deltaY < 0) {
                    if ($scope.scrollpos == 0) return
                    $scope.scrollpos-=speed
                }
        
                
                $rootScope.$broadcast('scrollpos',$scope.scrollpos)
                return false
           })


           $scope.redraw = function()  { 
                 if (!$scope.networks) return
                 
                 var leftaxis = 38
                 var rightmargin = 14
                 var botaxis = 30
                 var tabheight = 15 // part of topaxis
                 var topaxis = 30 + tabheight

                 var ctx = $scope.context
                 var canvas = $scope.canvas

                 //fill background 
                 ctx.fillStyle = "white";
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                 //draw title
                 ctx.fillStyle = "black";
                 ctx.font = "15px sans";
                 ctx.textAlign = "center";

                 var lookup = {
                    'm': 'Mutual Information',
                    'xx': 'XX Correlator',
                    'yy': 'YY Correlator',
                    'zz': 'ZZ Correlator',
                    'gxx': 'g2(XX)',
                    'gyy': 'g2(YY)',
                    'gzz': 'g2(ZZ)',
                 }

                 if ($scope.mode == 'arcs') ctx.fillText(lookup[$scope.key]+" Networks", canvas.width/2, tabheight+20); 
                 else if ($scope.mode == 'edge') ctx.fillText(lookup[$scope.key]+" Edge Distrib.", canvas.width/2, tabheight+20); 
                 else ctx.fillText(lookup[$scope.key]+" Degree Distrib.", canvas.width/2, tabheight+20); 

                 //draw tab
                 ctx.fillStyle = $scope.color
                 ctx.fillRect(0,0,canvas.width,tabheight)
                 ctx.fillStyle = "black";

                 // draw axis labels
                 ctx.font = "13px sans";
                 if ($scope.mode == 'arcs') ctx.fillText("Position",leftaxis-5 + (canvas.width-10-leftaxis)/2   , canvas.height-5); 
                 else if ($scope.mode == 'edge') ctx.fillText("Edge Weight",leftaxis-5 + (canvas.width-10-leftaxis)/2   , canvas.height-5);
                 else  ctx.fillText("Vertex Degree",leftaxis-5 + (canvas.width-10-leftaxis)/2   , canvas.height-5);

                 ctx.translate(15,canvas.height/2)
                 ctx.rotate(-Math.PI/2)
                 ctx.fillText("Iteration",0,0); 
                 ctx.rotate(Math.PI/2)
                 ctx.translate(-15,-canvas.height/2)

                 //determine sizes
                 var width = canvas.width - leftaxis - rightmargin
                 var height = canvas.height - topaxis - botaxis

                 var statelength = $scope.networks[0].length
                 if ($scope.mode != 'arcs') statelength = 10

                 var boxdim = width/statelength
                 var boxheight = 40
                 var numheight = (height - (height%boxheight))/boxheight


                 var boxx = function(j) { return leftaxis + j*boxdim  } 
                 var boxy = function(i) { return topaxis + (i-$scope.scrollpos)*boxheight } 

            

                 //draw numbers
                 ctx.fillStyle = "black"
                 ctx.font = "10px sans";
                 ctx.textAlign = "right";
                 for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                    ctx.fillText(i,leftaxis - boxdim/4,boxy(i)+boxheight/2+5); 
                 }

                 ctx.textAlign = "center";
                 for (var j = 0; j < statelength  ; j++) {
                    if ($scope.mode == 'arcs') {
                        ctx.fillText(j,boxx(j)+boxdim/2,canvas.height - botaxis + 3*boxdim/4); 
                    } else {
                        if (j == 0) ctx.fillText(j,boxx(j),canvas.height - botaxis + 2*boxdim/4); 
                        else ctx.fillText("."+j,boxx(j),canvas.height - botaxis + 2*boxdim/4); 
                    }
                 }

                if ($scope.mode != 'arcs') ctx.fillText(1,boxx(10),canvas.height - botaxis + 2*boxdim/4); 

                //underline
                 if ($scope.mode != 'arcs')  {
                     for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                         if ($scope.networks[i] === undefined) continue
                         ctx.strokeStyle = "black"
                         ctx.lineWidth = 1
                         ctx.beginPath()
                         ctx.moveTo(boxx(0),boxy(i+1)+0.5)
                         ctx.lineTo(boxx(statelength),boxy(i+1)+0.5)
                         ctx.stroke() 
                     }
                 }

                 //draw arcs
                 if ($scope.mode == 'arcs') { 
                     for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                         if ($scope.networks[i] === undefined) continue

                        for (var j = 0; j< statelength  ; j++) {
                            var dot = 2
                            ctx.fillRect(boxx(j)+boxdim/2-dot,boxy(i)+boxheight-dot,2*dot,2*dot)
                        }
                        for (var j = 0; j< statelength  ; j++) {
                            for (var k = 0; k < statelength  ; k++) {
                                if (j == k) continue
                                ctx.lineWidth = $scope.networks[i][j][k]*4

                                
                                var low = boxy(i) + boxheight*(1-(Math.abs(j-k)/statelength))
                                var high = boxy(i)+boxheight
                                var left = boxx(j)+boxdim/2 
                                var right = boxx(k)+boxdim/2 
                                
                                ctx.beginPath();
                                ctx.moveTo(left,high)
                                ctx.bezierCurveTo(left,low,right,low,right,high)
                                ctx.stroke();
                                  
                            }
                        }
                    }
                 } else {
                     var histmax = 0
                     var allhist = []
                     
                     for (var i = 0; i < $scope.networks.length;i++ ) {
                         if ($scope.networks[i] === undefined) continue

                         var hist = []
                         for (var l = 0; l < statelength; l++) hist[l] = 0

                         for (var j = 0; j< $scope.networks[0].length  ; j++) {
                            if ($scope.mode == 'edge')  {
                                for (var k = 0; k < $scope.networks[0].length  ; k++) {
                                    var value = $scope.networks[i][j][k]
                                    for (var l = 0; l < statelength; l++) {
                                        if ( value >= l/10 && value < (l+1)/10) hist[l]++
                                    }
                                    if (value == 1) hist[statelength-1]++
                                }
                            } else {
                                var value = 0
                                for (var k = 0; k < $scope.networks[0].length  ; k++) {
                                    value += $scope.networks[i][j][k]
           
                                }
                                value = value/$scope.networks[0].length
                                for (var l = 0; l < statelength; l++) {
                                    if ( value >= l/10 && value < (l+1)/10) {
                                        hist[l]++
                                    }
                                }
                                if (value == 1) hist[statelength-1]++
                            }
                         }
                         for (var l = 0; l < statelength; l++) if (hist[l] > histmax) histmax =  hist[l]
               
                         allhist.push(hist)

                     }


                     for (var i = $scope.scrollpos; i-$scope.scrollpos < numheight;i++ ) {
                         if ($scope.networks[i] === undefined) continue
                          for (var l = 0; l < statelength; l++) {
                             var boxw = 3
                             var boxh = 0.8*allhist[i][l]/histmax
                             ctx.fillRect(boxx(l)+boxdim/2-boxw,boxy(i)+(1-boxh)*boxheight,boxw*2,boxheight*boxh)
                         }
                     }    
                 }
 
                 
           }
           
           $scope.$watch(function() { return $scope.update; },$scope.redraw,true);
           $scope.$watch(function() { return $scope.mode; },$scope.redraw,true);
           $scope.$on('scrollpos',function(e,pos) {
               $scope.scrollpos = pos
               $scope.redraw()
           }, true)

            $scope.qbitcolor = function(density) {
               if (density[0][0].re + density[1][1].re - 1 > 1e-1) {
                   throw "matrix " +$scope.showDensity(density)+ " violates trace condition! ("+ JSON.stringify(density[0][0].re + density[1][1].re)   +"+" + JSON.stringify( density[0][0].im + density[1][1].im) + "i)"
               }

               var ph = math.complex(density[0][1]).toPolar().phi/Math.PI

               var r = 0
               var g = 0
               var b = 0
               if (math.complex(density[0][1]).toPolar().r > 1e-3)  {
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
                
                var maxmixed = false
                if (len < 1e-2) maxmixed = true


                return ["rgb("+Math.round(r)+","+Math.round(g)+","+Math.round(b)+")",maxmixed]
            }

          
    }}
});


