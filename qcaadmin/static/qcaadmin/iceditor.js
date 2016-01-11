
QCAAdmin.controller('ICEditor', ["$scope", "$rootScope",'$http','$timeout',  function($scope,$rootScope,$http,$timeout) {

    $scope.update = 0

    $scope.displays = {}
    $scope.toggle = function(idx) {
        if ($scope.displays[idx]) $scope.displays[idx] = false
        else $scope.displays[idx] = true
    } 

    $scope.length = 15
    $scope.chLength = function(dir) {
        if (dir == '+' && $scope.length < 24) $scope.length++
        if (dir == '-' && $scope.length > 6) $scope.length--
        if (Number.isInteger(dir)) $scope.length = dir

        for (var i = 0; i < $scope.stateData.length; i++) {
            while ($scope.stateData[i].values.length < $scope.length)  {
                $scope.stateData[i].values.push(0)
            }
            while ($scope.stateData[i].values.length > $scope.length)  {
                $scope.stateData[i].values.splice($scope.stateData[i].values.length-1,1)
            }
        }

        $rootScope.$broadcast('iclength',$scope.length)
        $scope.update++
    }

    $scope.title = ''


    $scope.canSaveIC = function() {
        if ($scope.title.length == 0) return false
        return true
    }

    $scope.saveIC = function() {
        $rootScope.displaymode = 'Sim'
        $rootScope.inspectedIC = false
    }

    $scope.stateData = []
    $scope.prepareState = function() {
        if ($scope.inspectedIC !== false && $rootScope.displaymode == 'IC') $scope.fetchICData()
        if ($rootScope.inspectedIC !== false || $rootScope.displaymode == 'Sim') return
        $scope.icdataset = false
        $scope.stateData = []
        $scope.stateData.push({
            "magnitude": 1,
            "phase": 0,
            "values": []
        })
        for (var i = 0; i < $scope.length; i++) $scope.stateData[0].values.push(0)
        $rootScope.updateStateNorm()
        $scope.title = ''
        $scope.update++
    }
    
    $scope.$watch(function() { return $rootScope.inspectedIC; },$scope.prepareState,true);
    $scope.$watch(function() { return $rootScope.displaymode; },$scope.prepareState,true);

    $scope.stateNorm = 1
    $rootScope.isAloneIC = function() {
        return $scope.stateData.length == 1
    }
    $rootScope.updateStateNorm = function() {
        var norm = 0 
        for (var i = 0; i < $scope.stateData.length; i++) {
            norm += $scope.stateData[i].magnitude*$scope.stateData[i].magnitude 
        }
        $scope.stateNorm = norm
        $rootScope.$broadcast('icnorm',norm)
        return norm
    }



    $rootScope.ICaction = function(idx,key) {
        if (key == 'Insert Above' || key == "Insert Below") {
            var newObj = {
                "magnitude": 1,
                "phase": 0,
                "values": []
            }
            for (var i = 0; i < $scope.length; i++) newObj.values.push(0)
            if (key == 'Insert Above') $scope.stateData.splice(idx,0,newObj)
            else $scope.stateData.splice(idx+1,0,newObj)

            if ($scope.stateData.length == 1) $scope.stateData[0].magnitude = 1
            $rootScope.updateStateNorm()
        }
        if (key == 'Clear Values') {    
            $scope.stateData[idx].values = []
            for (var i = 0; i < $scope.length; i++) $scope.stateData[idx].values.push(0)
        }

        if (key == 'Remove Comp.') {    
            if ($scope.stateData.length == 1) return
            $scope.stateData.splice(idx,1)
            if ($scope.stateData.length == 1) $scope.stateData[0].magnitude = 1
            $rootScope.updateStateNorm()
        }
        $scope.$apply(function() {
            $scope.update++
            $scope.outofdate = true
        })

    
    }

    $scope.$watch(function() { return $scope.update; },function() {
//        $scope.outofdate = true
    },true);

    $scope.$on('magnitude',function(e) {$scope.$apply(function() {$scope.outofdate = true})})
    $scope.$on('phase',function(e) {$scope.$apply(function() {$scope.outofdate = true})})
    $scope.$on('value',function(e) {$scope.$apply(function() {$scope.outofdate = true})})

    $scope.fetchingic = false
    $scope.outofdate = true
    $scope.icdataset = false
    $scope.fetchICData = function() {
        $scope.fetchingic = true

        var args = {}
        if ($scope.inspectedIC !== false) {
            args['pk'] = $rootScope.inspectedIC
        } else {
            args['compList'] = $scope.stateData
        }
        
        args['Password'] = $rootScope.password
        $http.post(window.prefix+'/getICData/',JSON.stringify(args), {}).then(function(response) {
            if (response.data == "Wrong password.") {
                    $rootScope.reqpass = true
                    $rootScope.password = ''
                    return
                }
            $scope.icdataset = response.data

            for (key in $scope.icdataset) {
                if (key == "title") continue
                $scope.icdataset[key] = JSON.parse($scope.icdataset[key])
            }
            
            $scope.bubsettings = {
                    "sites": "color",//'color', 'X', 'Y', 'Z', or 'none'
                    "siteent": true,
                    "cutent": true,
                }

            $scope.fetchingic = false
            $scope.outofdate = false
            $scope.update++
        },function(response) {
            document.write(response.data)
        })

    }

    $scope.saveIC = function() {
        var args = {}
        args['compList'] = $scope.stateData
        args['title'] = $scope.title
        args['Password'] = $rootScope.password
        $http.post(window.prefix+'/saveIC/',JSON.stringify(args), {}).then(function(response) {
            if (response.data == "Wrong password.") {
                    $rootScope.reqpass = true
                    $rootScope.password = ''
                    return
                }
            $rootScope.inspectedIC  = response.data
            $rootScope.$broadcast('newIC')
        },function(response) {
            document.write(response.data)
        })
    
    }

    $scope.bubsettings = {
        "sites": "color",//'color', 'X', 'Y', 'Z', or 'none'
        "siteent": true,
        "cutent": true,
    }



 }])


QCAAdmin.directive("staterow", function ($rootScope)
  {
    return {
        restrict: 'A',
        scope: {
            staterow: '=',
            norm: '=',
            update: '=',
            index: '=',
            outofdate: '=',
        },
        template: "<canvas width='500px' height='100px'></canvas>",
        link: function($scope, element, xattrs) {
            $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');
          
           $scope.normpos = 40
           $scope.comppos = 118
           
           $scope.bubpos = 190
           $scope.bubradius = 18

           $scope.topButtons = {
                "Insert Above":170,
                "Clear Values": 280,
           }
           $scope.botButtons = {
                "Insert Below":170,
                "Remove Comp.": 280,
           }
           
           var inrange = function(bx,dx,by,dy,e) {
                    //$scope.context.fillStyle = "black"
                    //$scope.context.fillRect(bx-dx,by-dy,2*dx,2*dy) 
                    if (e.offsetX < bx-dx) return false 
                    if (e.offsetX > bx+dx) return false 
                    if (e.offsetY < by-dy) return false 
                    if (e.offsetY > by+dy) return false 
                    return true
                } 


           element[0].addEventListener("wheel", function(e) {
                                
                var canvas = $scope.canvas
               if (inrange($scope.comppos,2*canvas.height/6,canvas.height/2,2*canvas.height/6,e)) {
                    e.stopPropagation()
                    e.preventDefault()
                    e.returnValue = false
                    
                    var speed = 1
                    if (e.deltaY < 0) $scope.staterow.phase += speed
                    if (e.deltaY > 0) $scope.staterow.phase -= speed
                    while ($scope.staterow.phase >= 16) $scope.staterow.phase -= 16
                    while ($scope.staterow.phase < 0) $scope.staterow.phase += 16
                
                    $scope.redraw()
                    $scope.update++
                    $rootScope.$broadcast('phase')
                    return false
               }

                if (inrange($scope.normpos,30,canvas.height/2,2*canvas.height/6,e)) {
                    if ($rootScope.isAloneIC()) return
                    e.stopPropagation()
                    e.preventDefault()
                    e.returnValue = false
                    
                    if (e.deltaY < 0 && $scope.staterow.magnitude > 1) $scope.staterow.magnitude -= 1
                    if (e.deltaY > 0) $scope.staterow.magnitude += 1
                    $scope.norm = $rootScope.updateStateNorm()
                    $rootScope.$broadcast('magnitude')
                    $scope.redraw()
                    $scope.update++
                    return false

                }
                
                return false
           })


           element[0].addEventListener("click", function(e) {
                  var bubpos = $scope.bubpos
                var bubradius = $scope.bubradius

                var canvas = $scope.canvas
                for (var i = 0; i< $scope.staterow.values.length; i++) {
                    if (inrange(bubpos + bubradius*2*i,bubradius,canvas.height/2,bubradius,e)) {
                        
                        $scope.staterow.values[i] =  1- $scope.staterow.values[i]
                        $rootScope.$broadcast('value')
                        $scope.redraw()
                        $scope.update++
                        return
                    }
                }

                for (key in $scope.topButtons) {
                    var pos = $scope.topButtons[key]
                    if (inrange(pos+50,50,15,8,e)) {
                        $rootScope.ICaction($scope.index,key)
                        $scope.redraw()
                        $scope.update++
                        return
                    }
                }
                for (key in $scope.botButtons) {
                    var pos = $scope.botButtons[key]
                    if (inrange(pos+50,50,canvas.height-15,8,e)) {
                        $rootScope.ICaction($scope.index,key)
                        $scope.redraw()
                        $scope.update++
                        return
                    }
                }

            })
    
           $scope.redraw = function()  { 
                var ctx = $scope.context
                var canvas = $scope.canvas

                // resize canvas
                var bubpos = $scope.bubpos
                var bubradius = $scope.bubradius
                canvas.width = bubpos + $scope.staterow.values.length*2*bubradius - bubradius/2

                //fill background 
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, canvas.width, canvas.height);

               //Draw buttons
                ctx.fillStyle = "black"
                ctx.font = "15px sans";
                for (key in $scope.topButtons) {
                    ctx.fillText(key,$scope.topButtons[key],20) 
                }
                for (key in $scope.botButtons) {
                    if (key == "Remove Comp." && $rootScope.isAloneIC()) continue
                    ctx.fillText(key,$scope.botButtons[key],canvas.height-10) 
                }





                //draw norm
                ctx.fillStyle = "black";
                ctx.strokeStyle = "black";
                ctx.font = "20px sans";
                ctx.lineWidth = 1.5
                ctx.textAlign = "center";
                var normpos = $scope.normpos
                if ($scope.norm == 1) {
                    ctx.fillText(1, normpos,canvas.height/2+8);
                } else {
                    ctx.fillText($scope.staterow.magnitude, normpos,canvas.height/2 - 10);
                    ctx.fillText($scope.norm, normpos+5,canvas.height/2 + 30);
                    ctx.beginPath()
                    ctx.moveTo(normpos-20,canvas.height/2+0.5)
                    ctx.lineTo(normpos+20,canvas.height/2+0.5)
                    ctx.stroke()

                    if ($scope.norm < 10) {
                        ctx.beginPath()
                        ctx.moveTo(normpos-15,canvas.height/2+20.5)
                        ctx.lineTo(normpos-12,canvas.height/2+20.5)
                        ctx.lineTo(normpos-9,canvas.height/2+30.5)
                        ctx.lineTo(normpos-8,canvas.height/2+8.5)
                        ctx.lineTo(normpos+10,canvas.height/2+8.5)
                        ctx.stroke()
                    } else {
                        ctx.beginPath()
                        ctx.moveTo(normpos-20,canvas.height/2+20.5)
                        ctx.lineTo(normpos-17,canvas.height/2+20.5)
                        ctx.lineTo(normpos-14,canvas.height/2+30.5)
                        ctx.lineTo(normpos-13,canvas.height/2+8.5)
                        ctx.lineTo(normpos+19,canvas.height/2+8.5)
                        ctx.stroke()
                    }

                
                }


                // Draw complex number picker
                ctx.lineWidth = 1
                var comppos = $scope.comppos
                
                var ph = $scope.staterow.phase/8
                if (ph > 1) ph -= 2
                var r = 0
                var g = 0
                var b = 0
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

                ctx.fillStyle = "rgb("+Math.round(r)+","+Math.round(g)+","+Math.round(b)+")"    
                ph = $scope.staterow.phase*Math.PI/8

                var radius = 2*canvas.height/6
                ctx.beginPath()
                ctx.arc(comppos,canvas.height/2,radius,0,Math.PI*2)
                ctx.fill()

                ctx.fillStyle= "black"


                ctx.beginPath()
                ctx.arc(comppos,canvas.height/2,radius,0,Math.PI*2)
                ctx.stroke()
                
                ctx.beginPath()
                 ctx.arc(comppos-radius - 13,canvas.height/2,3,0,Math.PI*2)
                ctx.fill()
                ctx.beginPath()
                 ctx.arc(comppos+radius + 13,canvas.height/2,3,0,Math.PI*2)
                ctx.fill()
              


                var basefactor = 0
                var text = ''
                if ($scope.staterow.phase == 0) text = '1'
                if ($scope.staterow.phase == 4) text = 'i'
                if ($scope.staterow.phase == 8) text = '-1'
                if ($scope.staterow.phase == 12) text = '-i'
                if (text != '') basefactor = 0.4

                if (text != '') {
                    ctx.fillStyle= "white"
                    ctx.beginPath()
                    ctx.arc(comppos,canvas.height/2,basefactor*radius,0,Math.PI*2)
                    ctx.fill()
                    ctx.fillStyle= "black"
                }
                if (text != '') ctx.fillText(text,comppos,canvas.height/2+7)


                ctx.lineWidth = 2
                ctx.beginPath()
                ctx.moveTo(comppos + radius*Math.cos(ph)*basefactor,canvas.height/2-radius*Math.sin(ph)*basefactor)
                ctx.lineTo(comppos + radius*Math.cos(ph),canvas.height/2-radius*Math.sin(ph))
                ctx.stroke()

                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.arc(comppos + radius*Math.cos(ph),canvas.height/2-radius*Math.sin(ph),5,0,Math.PI*2)
                ctx.fill()

                // Draw bubbles
              
                for (var i = 0; i< $scope.staterow.values.length; i++) {
                    var state = $scope.staterow.values[i]
                    ctx.fillStyle = "white"
                    if (state) ctx.fillStyle = "black"

                    ctx.beginPath()
                    ctx.arc(bubpos + bubradius*2*i,canvas.height/2,0.8*bubradius,0,Math.PI*2)
                    ctx.fill()


                    ctx.beginPath()
                    ctx.arc(bubpos + bubradius*2*i,canvas.height/2,0.8*bubradius,0,Math.PI*2)
                    ctx.stroke()

                
                }
                
                 

    
            }

           $scope.$watch(function() { return $scope.update; },$scope.redraw,true);
           $scope.$watch(function() { return $scope.norm; },$scope.redraw,true);
           $scope.$on('icnorm',function(e,norm) {
               $scope.norm = norm
               $scope.redraw()
           }, true)
             $scope.$on('iclength',function(e,length) {
               $scope.length = length
               $scope.redraw()
           }, true)


        }
    }
})


QCAAdmin.directive("bubble", function ($rootScope)
  {
    return {
        restrict: 'A',
        scope: {
            bubble: '=',
            settings: '=',
            update: '=',
        },
        template: "<canvas width='250px' height='80px'></canvas>",
        link: function($scope, element, xattrs) {
           $scope.canvas = element.find('canvas')[0];
           $scope.context = $scope.canvas.getContext('2d');


           $scope.redraw = function()  { 
                if ($scope.bubble["one_site"] == undefined) return
                 
                 var ctx = $scope.context
                 var canvas = $scope.canvas

             
                 
                 //determine sizes
                 var boxdim = 35
                 var statelength = $scope.bubble["one_site"][0].length

                 canvas.width = boxdim*(statelength+1)

                 var h = canvas.height/2
                 var boxx = function(j) { return (j+1)*boxdim } 
            

                    /*
                    $scope.bubsettings = {
                            "sites": "color"//'color', 'X', 'Y', 'Z', or 'none'
                            "siteent": true//site entropy
                            "cutent": true//site entropy
                        
                        }
                    */

               //fill background 
                 ctx.fillStyle = "white";
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
                 

                 //draw content
                 $scope.strokeStyle = "black"
                
                 if ($scope.settings.cutent) {
                        for (var j = 0; j+1 < statelength  ; j++) {
                            if (Math.abs($scope.bubble["sc"][0][j]*4) < 1e-3) continue
                            ctx.lineWidth = Math.abs($scope.bubble["sc"][0][j]*4)
                            ctx.beginPath();
                            ctx.moveTo(boxx(j),h)
                            ctx.lineTo(boxx(j+1),h)
                            ctx.stroke();
                              
                        }
                 }

                
                 var radius = boxdim/2
                 if (!$scope.settings.cutent) radius = boxdim/Math.sqrt(2)

                    for (var j = 0; j < statelength  ; j++) {

                        var density = $scope.bubble["one_site"][0][j]
                        if ($scope.settings.sites == 'color') { 
                            var bubble = $scope.qbitcolor(density)
                            if (!bubble[1]) ctx.fillStyle= bubble[0]
                            else {
                                var grd=ctx.createRadialGradient(boxx(j),h,0,boxx(j),h,radius);
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
                        }

                        if ($scope.settings.sites != 'none' ) { 
                            if ($scope.settings.cutent)  {
                                ctx.beginPath();
                                ctx.arc(boxx(j),h,4*boxdim/10,0,2*Math.PI);
                                ctx.fill();
                            } else ctx.fillRect(boxx(j)-boxdim/2,h-boxdim/2,boxdim,boxdim)
                        }

                        if (  !(bubble && bubble[1]) && $scope.settings.siteent ) {
                            if ($scope.settings.sites != 'none') {
                                var w = $scope.bubble["s"][0][j]*2.5
                                ctx.lineWidth = w 
                                if ($scope.settings.cutent)  {
                                    ctx.beginPath();
                                    ctx.arc(boxx(j),h,4*boxdim/10,0,2*Math.PI);
                                    ctx.stroke();
                                } else ctx.strokeRect(boxx(j)-boxdim/2+w/2,h-boxdim/2+w/2,boxdim-w/2,boxdim-w/2)
                            } else {
                                var w = $scope.bubble["s"][0][j]*4*boxdim/10
                                ctx.fillStyle = "black"
                                if ($scope.settings.cutent)  {
                                    ctx.beginPath();
                                    ctx.arc(boxx(j),h,w,0,2*Math.PI);
                                    ctx.fill();
                                } else ctx.fillRect(boxx(j)-w,h-w,w*2,w*2)
                            }
                        }


                 
                    }

                 if ($scope.settings.sites == 'none' && !$scope.settings.siteent && !$scope.settings.cutent) {
                    $scope.fillStyle = "black"
                     ctx.font = "15px sans";
                    ctx.fillText("Nothing Selected",canvas.width/2  , canvas.height/2); 
                 
                 }

                 
           }
           
           $scope.$watch(function() { return $scope.update; },$scope.redraw,true);
           $scope.$watch(function() { return $scope.settings; },$scope.redraw,true);
            
           $scope.qbitcolor = function(density) {

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

