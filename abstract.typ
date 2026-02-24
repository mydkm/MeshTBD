#import "@preview/axiom:0.1.0": *
#set page(
  numbering: "1",
)
#set document(
  title: [Automated Generation of 3D-Printable Custom Cast  
Geometry From Noisy Limb Scan Meshes]
)
#show columns: set text(size: 10pt)
#set heading(numbering: "1.") 
#show heading: set text(size: 10pt, weight: "bold")
#show figure: set block(below: (2.0em))
#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)

#let date = datetime.today()
#show table.cell.where(y: 0): set text(weight: "bold")


// Document Contents
#title()
#v(1.5em, weak: true)
*Joshua Davidov*#super[1], Ahikara Sandrasagra#super[2], Mili Shah#super[3] \
(1) Department of Mechanical Engineering, The Cooper Union, New York, NY, USA \ 
(2) Department of Electrical Engineering, The Cooper Union, New York, NY, USA \ 
(3) Department of Mathematics, The Cooper Union, New York, NY, USA \ 

#columns(2,
         gutter: 12pt)[
= Abstract <abstract>

3D scanning enables rapid analysis of patient-specific limb geometry for custom medical casts. but the resulting 3D meshes are rarely fabrication-ready. Typical scans contain holes, self-intersections, non-manifold edges, disconnected fragments, and inconsistent triangulation, which can make manual CAD remodeling time-consuming. Additionally, cast design requires controlled clearance, thickness, and trimming that preserve anatomical fit while enabling ventilation and fastening features.

This work presents an open-source Python pipeline, MeshTBD, that converts an input limb scan into a watertight 3D model of a printable cast. The workflow integrates commonly used mesh processing and visualization toolchains (e.g., PyMeshLab, Blender Python API, PyVista) to (1) import and standardize heterogeneous scan formats, (2) perform mesh hygiene and repair, (3) extract and stabilize a usable limb surface suitable for downstream geometric operations, and (4) generate a cast geometry through surface offsetting/thickening and trimming operations. The pipeline produces a closed, manufacturable cast model that can be exported directly for additive manufacturing workflows.

The ability to reliably transform limb scans via MeshTBD is a first step towards reproducible breathable and water-friendly casts.

= Plane-fitting Algorithm (DRAFT) <setup>

 Let $S subset RR^(3)$ be a (piecewise smooth) closed surface that bounds a solid region $Omega $.

//  We want a plane $Pi $ that

//   +  passes through the *volume centroid* $c $ of $Omega $,
//   +  contains the “radial” direction from $c $ to the surface (your arrow),
//   +  is “transverse” in the sense of being *approximately perpendicular to the object’s long axis*.

 /* \hrule */

 Define the volume and centroid of $Omega $: 
 #numbered_eq($  V = integral_(Omega) d V $)<eq-1>
 #numbered_eq($ c = 1/V integral_(Omega) x d V $)<eq-2>

 For an oriented triangle mesh with faces $(bold(v_0),bold(v_1),bold(v_2))$, compute these using signed tetrahedra about a reference point $bold(r) in RR^(3)$ (often $bold(r) = 0 $ or the mesh’s bounding-box center):

  -  Signed volume of tetra $(bold(r),bold(v_(0)),bold(v_(1)),bold(v_(2)))$: 
  #numbered_eq($  V_(i)= 1/6(bold(v_(0))- bold(r)) dot ((bold(v_(1))- bold(r))times(bold(v_(2))- bold(r)))  $)<eq-3>

  -  Centroid of that tetra: 
  #numbered_eq($  bold(c_(i))= frac(bold(r) + bold(v_(0))+ bold(v_(1))+ bold(v_(2)), 4)  $)<eq-4>

  -  Total signed volume and centroid: 
  #numbered_eq($  V = sum_(i) V_(i) $)<eq-5> 
  #numbered_eq($ bold(c) = 1/V sum_(i) V_(i) bold(c_(i))  $)<eq-6>

 Note: If the mesh is watertight and consistently oriented, $V != 0 $ and this matches the enclosed solid’s centroid.

 /* \hrule */

 Then, choose the surface point $bold(p_c)$, the closest point on $S $ to $bold(c)$: 
 #numbered_eq($  bold(p_c) in argmin_(x in S)norm(bold(x)-bold(c))^2 $)<eq-7>

 Define the arrow direction as: 
 #numbered_eq($  bold(hat(n)) = frac(bold(p_c - c), bold(norm(p_c-c)))  $)<eq-8>

 /* \hrule */


 Let ${x_(k)}_(k = 1)^(N)$ be a set of points located on $S$. Define the centered covariance matrix: 
 #numbered_eq($  Sigma = 1/N sum_(k = 1)^(N)(bold(x_(k)- c))(bold(x_(k)- c))^(top)  $)<eq-9>

 Note that $Sigma$ is real symmetric. Thus, by spectral theorem, $Sigma$ is orthogonal and has 3 eigenpairs.
 
 Let $(lambda_1,bold(u_1)),(lambda_2,bold(u_2)),(lambda_3,bold(u_3)$ be eigenpairs of $Sigma $ with ($lambda_(1)>= lambda_(2)>= lambda_(3)>= 0  $)
  and $bold(u_(1)),bold(u_(2)),bold(u_(3))$ orthonormal. Define the principal axis as: 
  #numbered_eq($  u : = u_(1)  $)<eq-10>

//  In this case, $u $ is the direction of maximum spread of the point cloud—usually the limb’s “length” direction.

 /* \hrule */


 Recall that a unit vector $bold(hat(n))$ is tangent to a plane $Pi $ iff its normal $bold(m) $ satisfies #numbered_eq($  bold(m) dot bold(n) = 0   $)<eq-11>

 Among all unit normals $bold(m) $ satisfying (11), we choose the one closest to the long axis $u $. This can be formalized to the following the constrained maximization problem: 
 #numbered_eq($  (max_(bold(m) in RR^(3))(bold(m) dot bold(u)) "s.t"  ((norm(bold(v)) = 1) and (bold(m) dot bold(n) = 0)))  $)<eq-12>

 The solution to this problem is exactly the normalized projection of $u $ onto the orthogonal complement of $n $ (negative triple product): 
 

//  *Geometric meaning:*

//   - $m $ is the direction “most aligned with the long axis” while still being orthogonal to the arrow direction $n $.
//   -  Since $m perp n $, the arrow direction lies in the plane.
//   -  Since $m approx u $, the plane is approximately perpendicular to the long axis (transverse slice).

 /* \hrule */

 A proof of this solution is available in @appendix[Appendix 1]. This algorithm is implemented into ```python planeCalc.py```, which is currently accessible in the #link("https://github.com/mydkm/MeshTBD")[MeshTBD Github repository].
 
 Finally, the plane through the centroid with normal $m $ is 
 #numbered_eq($  Pi = {x in RR^(3): m (x - c)= 0 }  $)

 Equivalently, in $(a x_1 + b x_2 + c x_3 + d = 0) $ form: $ (a,b,c)= m $ 
 #numbered_eq($ d = - (m dot c)  $)

 /* \hrule */

 Do note, if $u $ is nearly parallel to $n $, then 
 #numbered_eq($  u -(u dot n)n approx 0  $) 
 and the “closest-to-$u $” condition doesn’t pick a unique direction (the transverse plane can spin freely while still containing $n $). In that case, the script tries $u_(2)$, then $u_(3)$, then a fixed world axis, always using the same projection formula.
 
 For each mesh vertex $x in S$, where $x=(x_i,y_i,z_i)$, compute:
 #numbered_eq($ s_i A $)

 This algorithm is not fully implemented into the main workflow present in ```python VoronoiFinal.py```, as (1) current scalar field assignment algorithms do not appropiately assign values to vertices in the thumb and (2) attempting to generate Voronoi cells with a Poisson Sampling incorrectly generates cells over the fingers in 3D hand scans. Variations of this algorithm with user input, similar to its use in the scale factor calculation, are currently under consideration.

 = Appendix <appendix>
 
 = References <references>
]

