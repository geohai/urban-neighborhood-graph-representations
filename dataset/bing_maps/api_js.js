var bounds = new Microsoft.Maps.LocationRect(new Microsoft.Maps.Location(28.332823, -81.492279), 0.01, 0.01);
Microsoft.Maps.Map.getClosestPanorama(bounds, onSuccess, onMissingCoverage);
function onSuccess(panoramaInfo) {
    var map = new Microsoft.Maps.Map(document.getElementById('myMap'), {
        zoom: 18,
        mapTypeId: Microsoft.Maps.MapTypeId.streetside,
        streetsideOptions: {
            panoramaInfo: panoramaInfo,
            onSuccessLoading: function () { return document.getElementById('printoutPanel').innerHTML = 'Streetside loaded'; },
            onErrorLoading: function () { return document.getElementById('printoutPanel').innerHTML = 'Streetside failed to loaded'; }
        }
    });
}
function onMissingCoverage() {
    document.getElementById('printoutPanel').innerHTML = 'No streetside coverage/error in getting the panorama';
}
