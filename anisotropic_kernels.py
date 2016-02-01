import numpy
import numpy.linalg
import scipy
import scipy.spatial


kernel = hou.ch( "kernel_radius" )
search = hou.ch( "search_radius" )
threshold = hou.ch( "threshold_constant" )
ks = hou.ch( "scaling_factor" )
kr = hou.ch( "eigenvalues_ratio" )


node = hou.pwd()
geo = node.geometry()
geo.clear()

particles = node.inputs()[ 0 ].geometry()
sphere = node.inputs()[ 1 ].geometry()


data = []
for particle in particles.points():
    data.append( numpy.array( particle.position() ) )

kdtree = scipy.spatial.KDTree( data )


for particle in particles.points():
    pos = numpy.array( particle.position() )
    
    ids = kdtree.query_ball_point( pos, search )
    
    closest_particles_num = len( ids )
    
    anisotropy = numpy.matrix([ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,1.0,0.0] ])

    if closest_particles_num > 0:
        weighted_mean = numpy.array([ 0, 0, 0 ])
        weighted_position = numpy.array([ 0, 0, 0 ])
        weight = 0
        weighting_function = 0
        
        for idx in ids:
            mag = numpy.linalg.norm( data[ idx ] - pos )
            weight = 1 - pow( ( mag / search ), 3 )
            weighting_function += weight
            weighted_position += data[ idx ] * weight
          
        weighted_mean = weighted_position / weighting_function
          
        covariance = numpy.matrix([ [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0] ])
          
        for idx in ids:
            mag = numpy.linalg.norm( data[ idx ] - pos )
            weight = 1 - pow( ( mag / search ), 3 )
            weighted_distance = data[ idx ] - weighted_mean

            covariance += weighted_distance * weighted_distance.reshape( 3, 1 ) * weight
            
        covariance = covariance / weighting_function
            
        R, diag, RT = numpy.linalg.svd( covariance )
      
        if closest_particles_num > threshold:
            diag[ 1 ] = max( diag[ 1 ], diag[ 0 ] / kr )
            diag[ 2 ] = max( diag[ 2 ], diag[ 0 ] / kr )
            diag *= ks
        else:
            diag = numpy.array( 1.0, 1.0, 1.0 )
        
            
        houR = hou.Matrix3([ R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2] ]).inverted()
        houDiag = hou.Matrix3([ diag[0], 0.0, 0.0, 0.0, diag[1], 0.0, 0.0, 0.0, diag[2] ])

        anisotropy = hou.Matrix4( houR * houDiag * houR.transposed() * ( 1 / kernel ) )
        
    ellipsoid = hou.Geometry()
    ellipsoid.merge( sphere )
    
    for point in ellipsoid.points():
        point.setPosition( ( point.position() * anisotropy ) + particle.position() )
        
    geo.merge( ellipsoid )