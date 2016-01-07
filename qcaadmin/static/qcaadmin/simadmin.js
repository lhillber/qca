var Flashcards = angular.module('QCAAdmin', []);

Flashcards.filter('unsafe', function($sce) {
    return function(val) {
        if (val.indexOf("<b>") > -1) {
            return $sce.trustAsHtml(val.replace(/\s+/g,''));
        }
        return $sce.trustAsHtml(val);
    };
});





Flashcards.controller('flashcards', ['$scope','$timeout','$http',function($scope,$timeout,$http) {
    $scope.fromLang = "pinyin"
    $scope.toLang = "english"
    $scope.reqText = ""
    $scope.toTexts = [""]
    $scope.fromText = ""
    $scope.otherTexts = [""]

    $scope.flipped = "noflip"

    $scope.length = 0
    $scope.previous = []
    $scope.current = []
    $scope.nextdate = ''
    $scope.timeuntil = ''
    $scope.message = ''
    $scope.chapters = []
    $scope.selected = 'None'
    $scope.quizonly = false
    $scope.onquiz = false

    $scope.error = ''

    $scope.fetch = function() {
        var append = ""
        if ($scope.selected != "None") {
            append = "&filter="+$scope.selected
        }
        if ($scope.quizonly) {
            append = append + "&quiz"
        }


        $http.get('/flashcards/getItems/?prev='+JSON.stringify($scope.previous)+append, {}).then(function(response) {
            if (response.data.indexOf === undefined) {

                if ($scope.previous.length > 5) $scope.previous.shift()
                $scope.current = [] 
                for (var i = 0; i < response.data.current.length; i++) {
                    $scope.current.push(JSON.parse(response.data.current[i])[0])
                    $scope.previous.push($scope.current[i].fields.triple)
                }

                $scope.length = response.data.length
                $scope.message = response.data.message
                $scope.chapters = response.data.chapters
                $scope.chapters.unshift("None")
                $scope.init() 
            } else {
                $scope.length = 0
                $scope.current = {}
                $scope.message = ''
                $scope.nextdate =  Date.parse(JSON.parse(response.data))
                $scope.countdownloop() 
            }


        },function(response) {
            $scope.error = response.statusText
        });
    }
    $scope.fetch()


    $scope.countdownloop = function() {
        var nexttime = $scope.nextdate - Date.now()
        if (nexttime < 0) nexttime = 0
        var seconds = nexttime/1000
        var minutes = seconds/60
        var hours = minutes/60
        var days = hours/24


        $scope.timeuntil = "等待: " + Math.floor(days) + "天 "
        $scope.timeuntil = $scope.timeuntil + Math.floor(hours%24) + "时 "
        $scope.timeuntil = $scope.timeuntil + Math.floor(minutes%60) + "分 "
        $scope.timeuntil = $scope.timeuntil + Math.floor(seconds%60) + "秒 "
    
        if (nexttime > 1000 && Math.floor(seconds%60) != 0) $timeout($scope.countdownloop,1000)
        else $scope.fetch()
    }

    $scope.init = function() {
        if ($scope.length == 0) return
        $scope.clear()
        var first = $scope.current[0]

        var dir = first.fields.direction
        var lookup = {
            'C': "characters",
            'E': "english",
            'P': "pinyin",
        }

        var other = "CEP".replace(dir[0],"").replace(dir[1],"")

        $scope.fromLang = lookup[dir[0]]
        $scope.toLang = lookup[dir[1]]
        $scope.otherLang = lookup[other]
        
        var lookup2 = {
            'C': "转换成字符：",
            'E': "Translate to English:",
            'P': "Zhuǎnhuàn chéng pīnyīn:",
        }

        $scope.reqText = lookup2[dir[1]]

        var triples = []
        for (var i = 0; i < $scope.current.length; i++) {
            triples.push($scope.current[i].fields.triple)
        }


        $http.get('/flashcards/getTriples/?pks='+JSON.stringify(triples), {}).then(function(response) {

            $scope.fromText = JSON.parse(response.data[0])[0].fields[lookup[dir[0]]]
            $scope.onquiz = JSON.parse(response.data[0])[0].fields.quiz

            $scope.toTexts = []
            $scope.otherTexts = []
            for (var i = 0; i < $scope.current.length; i++) {
                var row = JSON.parse(response.data[i])
                $scope.toTexts.push(row[0].fields[lookup[dir[1]]])
                $scope.otherTexts.push(row[0].fields[lookup[other]])
            }

        },function(response) {
            $scope.error = response.statusText
        });
        
    }


    $scope.ready= function() {
        $scope.flipped = "flip"
    }

    $scope.correct= function() {
        $scope.toTexts = ["..."]
        $scope.fromText = "..."
        $scope.otherTexts = ["..."]
        $scope.flipped = "noflip"

        $http.get('/flashcards/submit/?correct&pk='+$scope.current[0].pk, {}).then(function(response) {
            $timeout($scope.fetch,100)
        },function(response) {
            $scope.error = response.statusText
        });
    }

    $scope.notcorrect = function() {
        $scope.toTexts = ["..."]
        $scope.fromText = "..."
        $scope.otherTexts = ["..."]
        $scope.flipped = "noflip"

        $http.get('/flashcards/submit/?pk='+$scope.current[0].pk, {}).then(function(response) {
            
            $timeout($scope.fetch,100)
        },function(response) {
            $scope.error = response.statusText
        });

    }

}])


Flashcards.directive("drawingpad", function() {
    return {
        template: "<canvas width='200' height='200'></canvas>",      
        link: function($scope, $element, $attrs) {
            $scope.canvas = $element.find('canvas')[0];
            $scope.context = $scope.canvas.getContext('2d');
            
            $scope.canvas.width = $element[0].offsetWidth - 10

            $scope.clear = function() {
                 var ctx = $scope.context
                 ctx.clearRect(0, 0, $scope.canvas.width, $scope.canvas.height);
                 ctx.fillStyle = "black";
                 ctx.font = "15px sans";
                 ctx.textAlign = "center";
                 ctx.fillText("Draw here", $scope.canvas.width/2, 15); 
                 ctx.fillText("Clear", 20, 15); 
            }
            $scope.$watch(function() { return $scope.toText },$scope.clear)

            $scope.dragging = false
            var down = function(e) {
                if (!e.offsetX && e.changedTouches && 0 in e.changedTouches) {
                    var xpos = e.changedTouches[0].pageX- $element[0].offsetLeft
                    var ypos = e.changedTouches[0].pageY - $element[0].offsetTop 
                    if (xpos < 40 && ypos < 20) $scope.clear()
                    else {
                        $scope.dragging = true
                        $scope.context.beginPath()
                        $scope.context.moveTo(xpos,ypos)
                    }
                } else {
                    if (e.offsetX < 40 && e.offsetY < 20) $scope.clear() 
                    else {
                        $scope.dragging = true
                        $scope.context.beginPath()
                        $scope.context.moveTo(e.offsetX,e.offsetY)
                    }
                }
                return false
            }

            var move = function(e) {
                if (!$scope.dragging) return 
                var ctx = $scope.context
                ctx.fillStyle = "black";
                //ctx.beginPath()
                if (e.offsetX) ctx.fillRect(e.offsetX-3,e.offsetY-3,6,6)
                else ctx.fillRect(e.changedTouches[0].pageX- $element[0].offsetLeft-3,e.changedTouches[0].pageY - $element[0].offsetTop-3,6,6)
                //ctx.fill()
                if (e.offsetX) ctx.lineTo(e.offsetX,e.offsetY)
                else ctx.lineTo(e.changedTouches[0].pageX- $element[0].offsetLeft,e.changedTouches[0].pageY - $element[0].offsetTop)
                                
                return false
            }

            var up = function() {
                if (!$scope.dragging) return
                $scope.dragging = false
                $scope.context.strokeStyle = "black"
                $scope.context.lineWidth = 8
                $scope.context.lineCap = "round"
                $scope.context.stroke()

                return false
            }

            $scope.canvas.onmousedown = down
            $scope.canvas.onmousemove = move
            $scope.canvas.onmouseup = up
            $scope.canvas.onmouseleave = up

            
            $scope.canvas.ontouchstart = down
            $scope.canvas.ontouchmove = move
            $scope.canvas.ontouchend = up
            $scope.canvas.ontouchleave = up
            //$scope.canvas.ontouchcancel = up
            
        }
    }
})



