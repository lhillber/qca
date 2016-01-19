var QCAAdmin = angular.module('QCAAdmin', []);

window.prefix = ''
if (window.location.href.includes('prallcase.caltech.edu/qca')) window.prefix = '/qca'

function readCookie(name) {
        var nameEQ = name + "=";
        var ca = document.cookie.split(';');
        for(var i=0;i < ca.length;i++) {
            var c = ca[i];
            while (c.charAt(0)==' ') c = c.substring(1,c.length);
            if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
        }
        return null;
    }

QCAAdmin.config(['$httpProvider', function($httpProvider) {

    $httpProvider.defaults.headers.post['X-CSRFToken'] = readCookie("csrftoken")
}]);


QCAAdmin.controller('qcaadmin', ['$scope','$timeout','$http','$rootScope',function($scope,$timeout,$http,$rootScope) {
    $scope.ics = []
    $scope.selectedics = []
    $scope.selectedictitles = []
    $scope.icfilter = ""
   
    $rootScope.rules = [102,150]
    $scope.Vs = ["X", "H", "HX", "HT", "HXT"]
    $rootScope.modes = ['block','sweep','alt']
    $scope.modedisplay = {
        'block':'Blk',
        'sweep':'Swp',
        'alt':'Alt'
    }
    $scope.cols = []
    $rootScope.makecols = function() {
        cols = []
        for (var j = 0; j < $rootScope.rules.length; j++) {
            for (var i = 0; i < $rootScope.modes.length; i++) {
                cols.push({
                    "rule": $rootScope.rules[j],
                    "mode": $rootScope.modes[i],
                })
            } 
        }
        $scope.cols = cols
    }
    $rootScope.makecols()

    $scope.sims = {}
    $rootScope.selectedsims = []

    $scope.simsrunning = []
    $scope.simupdateflag = false
    $scope.error = ""
    $scope.reload = function() {window.location.reload()}

    $rootScope.reqpass = false
    $rootScope.password = ''

    $rootScope.displaymode = 'Sim'
    $rootScope.inspectedIC = false
    $scope.addIC = function() {
        $rootScope.displaymode = 'IC'
        $rootScope.inspectedIC = false
    }
    $scope.inspectIC = function(e,pk) {
        e.stopPropagation()
        $rootScope.displaymode = 'IC'
        $rootScope.inspectedIC = pk
    }
   
    $rootScope.$on('newIC',function() {
        $scope.fetchICS()
        })
    $scope.fetchICS = function(extraic,extrasim) {
        $http.get(window.prefix+'/iclist/?filter='+$scope.icfilter, {}).then(function(response) {
            $scope.ics = response.data
            $scope.fetchSimList(extraic,extrasim)
        },function(response) {
            $scope.error = response.statusText
        })
    }
    $scope.fetchICS()
   

    $scope.clickIC = function(index) {
            ic = $scope.ics[index]
            var selidx = $scope.selectedics.indexOf(ic.pk)
            if (selidx == -1) {
                $scope.selectedics.push(ic.pk)
                $scope.selectedictitles.push(ic.title + " ("+ic.length+" sites)")
            }
            else {
                $scope.selectedics.splice(selidx,1)
                $scope.selectedictitles.splice(selidx,1)
            }
            $scope.fetchSimList()
    }

    $scope.fetchSimList = function(extraic,extrasim) {

         var callback = function(title) {
            return function(response) {
                $scope.sims[title] = response.data
                if (extraic !== undefined && extraic == title) {
                    $scope.sims[title].push(extrasim)
                    if ($scope.simsrunning.indexOf(extrasim.pk) == -1) $scope.simsrunning.push(extrasim.pk)
                    $scope.simupdateflag = true
                    for (var i = 0; i < $scope.ics.length; i++) {
                        if ($scope.ics[i].pk == extrasim.ic) $scope.ics[i].simrunning = true
                    }
                    $scope.delayfetch = true
                }
            }
         } 

         var titles = []
         for (var i = 0; i < $scope.selectedics.length; i++) {
            var title = $scope.selectedictitles[i]
            titles.push(title)

            $http.get(window.prefix+'/datalist/?ic='+$scope.selectedics[i], {}).then(callback(title),
            function(response) {
                $scope.error = response.statusText
            })
         }

         for (var key in $scope.sims) {
            if ($scope.sims.hasOwnProperty(key) && titles.indexOf(key) == -1) {
                delete $scope.sims[key]
            }
         }
    }

    $scope.simClass = function(icname,V,rule,mode) {
        for (var i = 0; i < $scope.sims[icname].length; i++) {
            var sim = $scope.sims[icname][i]
            if (sim.V == V && sim.R == rule && sim.mode == mode) {
                if (sim.completed) return ""
                else return "computing"
            }
        
        }
        return "uncomputed"    
    }


    $scope.simColor = function(icname,V,rule,mode) {
        for (var i = 0; i < $scope.sims[icname].length; i++) {
            var sim = $scope.sims[icname][i]
            if (sim.V == V && sim.R == rule && sim.mode == mode) {
                return {'background': $rootScope.colorforsim(sim.pk)}
            }
        
        }
        return {'background': ''}
    }

    $scope.startSim = function(icname,V,rule,mode) {

        ic = $scope.selectedictitles.indexOf(icname)

        for (var i = 0; i < $scope.sims[icname].length; i++) {
            var sim = $scope.sims[icname][i]
            if (sim.V == V && sim.R == rule && sim.mode == mode) {
                return 
            }
        }


        if ($rootScope.password == '') {
            $rootScope.reqpass = true
            return
        }

        if ($scope.numsimsrunning == 4) return
        var args = {
            'X-CSRFToken': readCookie("csrftoken"),
            'Password':$rootScope.password,
            "R":rule,
            "V":V,
            "IC":$scope.selectedics[ic],
            "mode":mode,
        }
        

        $http.post(window.prefix+'/startSimulation/?',JSON.stringify(args), {}).then(function(response) {
            if (response.data != "Launched") {
                if (response.data == "Wrong password.") {
                    $rootScope.reqpass = true
                    $rootScope.password = ''
                    return
                }

                $scope.error = response.data
                return
            }
            console.log("Started simulation")
            //$scope.sims[$scope.selectedictitles[ic]].push(

            $scope.fetchICS($scope.selectedictitles[ic],{
                "V": V,
                "R": rule,
                "mode": mode,
                "completed": false,
                "ic":$scope.selectedics[ic]
            })
        },function(response) {
            $scope.error = response.statusText
            document.write(response.data)
        })

    }

    $scope.simUpdate = function(disable) {
        if ($scope.delayfetch) {
            $scope.delayfetch = false
            $timeout($scope.simUpdate,5000)
        }
        $http.get(window.prefix+'/simStatus/', {}).then(function(response) {
            if (JSON.stringify($scope.simsrunning) != JSON.stringify(response.data.sort()) || $scope.simupdateflag) {
                $scope.simupdateflag = false
                $scope.simsrunning = response.data.sort()
                $scope.fetchICS()
            }

            if (disable === undefined) $timeout($scope.simUpdate,5000)
        },function(response) {
            $scope.error = response.statusText
        })
    }
    $scope.simUpdate()
    $scope.delayfetch = false


    
    $rootScope.colorforsim = function(pk) {
        var i = $rootScope.selectedsims.indexOf(pk) 
        if (i == -1) return ""
        var ph = i*2/$rootScope.selectedsims.length

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
    
        return "rgb("+Math.round(r)+","+Math.round(g)+","+Math.round(b)+")"
    }

    $scope.selectSim = function(icname,V,rule,mode) {
         for (var i = 0; i < $scope.sims[icname].length; i++) {
            var sim = $scope.sims[icname][i]
            if (sim.V == V && sim.R == rule && sim.mode == mode) {
                if (!sim.completed) return
                var idx = $rootScope.selectedsims.indexOf(sim.pk)
                if (idx == -1) $rootScope.selectedsims.push(sim.pk)
                else $rootScope.selectedsims.splice(idx,1)
                $rootScope.selectedsims.sort()
            
                $rootScope.displaymode = 'Sim'
                $rootScope.inspectedIC = false
                return
            }
        
        }
    }

}])




QCAAdmin.controller('modeselect', ['$scope','$timeout','$http','$rootScope',function($scope,$timeout,$http,$rootScope) {
    $scope.allrules = [204,201,198,195,156,153,150,147,108,105,102,99,60,157,57,51]
    $scope.allmodes = ['block','sweep','alt']

    $scope.setMode = function(mode) {
        var idx = $rootScope.modes.indexOf(mode)
        if (idx > -1) {
            if ($rootScope.modes.length == 1) return
            $rootScope.modes.splice(idx,1)
        } else {
            if (($rootScope.modes.length+1) * $rootScope.rules.length > 6) return
            var newmodes = []
            for (var i = 0; i < $scope.allmodes.length; i++) {
                if ($rootScope.modes.indexOf($scope.allmodes[i]) > -1 || $scope.allmodes[i] == mode) {
                    newmodes.push($scope.allmodes[i])
                }
            }
            $rootScope.modes = newmodes
        }
        $rootScope.makecols()
    }
    
    $scope.setRule = function(rule) {
        var idx = $rootScope.rules.indexOf(rule)
        if (idx > -1) {
            if ($rootScope.rules.length == 1) return
            $rootScope.rules.splice(idx,1)
        } else {
            if (($rootScope.rules.length+1) * $rootScope.modes.length > 6) return
            var newrules = []
            for (var i = 0; i < $scope.allrules.length; i++) {
                if ($rootScope.rules.indexOf($scope.allrules[i]) > -1 || $scope.allrules[i] == rule) {
                    newrules.push($scope.allrules[i])
                }
            }
            $rootScope.rules = newrules
        }
        $rootScope.makecols()
    }
    

}])
