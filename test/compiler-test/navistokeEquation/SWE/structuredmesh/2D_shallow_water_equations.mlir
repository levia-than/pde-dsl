module {
  // 计算单点通量
  func.func @compute_point_flux(%h: f32, %m: f32, %g: f32) -> (f32, f32) {
    %m_sq = arith.mulf %m, %m : f32
    %h_sq = arith.mulf %h, %h : f32
    %m_div_h = arith.divf %m_sq, %h : f32
    %g_half = arith.divf %g, %c2 : f32
    %gh_sq = arith.mulf %g_half, %h_sq : f32
    
    // 返回 (M, M^2/H + gH^2/2)
    return %m, %m_div_h_plus_gh : f32, f32
  }

  // 计算网格间通量
  func.func @compute_flux(
    %h_l: f32, %m_l: f32,
    %h_r: f32, %m_r: f32,
    %g: f32, %d: f32
  ) -> (f32, f32) {
    // 计算左右通量
    %fl:2 = call @compute_point_flux(%h_l, %m_l, %g) : (f32, f32, f32) -> (f32, f32)
    %fr:2 = call @compute_point_flux(%h_r, %m_r, %g) : (f32, f32, f32) -> (f32, f32)
    
    // 数值粘性项
    %h_diff = arith.subf %h_l, %h_r : f32
    %m_diff = arith.subf %m_l, %m_r : f32
    %d_half = arith.mulf %d, %c0_5 : f32
    %dh = arith.mulf %d_half, %h_diff : f32
    %dm = arith.mulf %d_half, %m_diff : f32
    
    // 计算平均通量
    %f1_avg = arith.addf %fl#0, %fr#0 : f32
    %f2_avg = arith.addf %fl#1, %fr#1 : f32
    %f1 = arith.divf %f1_avg, %c2 : f32
    %f2 = arith.divf %f2_avg, %c2 : f32
    
    // 加入数值粘性
    %f1_final = arith.addf %f1, %dh : f32
    %f2_final = arith.addf %f2, %dm : f32
    
    return %f1_final, %f2_final : f32, f32
  }

  // RK4单步计算
  func.func @rk4_step(
    %h: memref<?xf32>, %m: memref<?xf32>,
    %h_new: memref<?xf32>, %m_new: memref<?xf32>,
    %dt: f32, %dx: f32, %g: f32, %d: f32
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %div = memref.dim %h, %c0 : memref<?xf32>
    
    // 分配k1-k4数组
    %k1_h = memref.alloc(%div) : memref<?xf32>
    %k1_m = memref.alloc(%div) : memref<?xf32>
    // ... k2,k3,k4类似
    %k2_h = memref.alloc(%div) : memref<?xf32>
    %k2_m = memref.alloc(%div) : memref<?xf32>

    %k3_h = memref.alloc(%div) : memref<?xf32>
    %k3_m = memref.alloc(%div) : memref<?xf32>

    %k4_h = memref.alloc(%div) : memref<?xf32>
    %k4_m = memref.alloc(%div) : memref<?xf32>
    
    // 计算k1
    scf.for %i = %c0 to %div step %c1 {
        %ip1 = arith.addi %i, %c1 : index
        %im1 = arith.subi %i, %c1 : index

        %h_i = memref.load %h[%i] : memref<?xf32>
        %m_i = memref.load %m[%i] : memref<?xf32>
        %h_ip1 = memref.load %h[%ip1] : memref<?xf32>
        %m_ip1 = memref.load %m[%ip1] : memref<?xf32>
        %h_im1 = memref.load %h[%im1] : memref<?xf32>
        %m_im1 = memref.load %m[%im1] : memref<?xf32>
      
        %flux_l:2 = func.call @compute_flux(%h_im1, %m_im1, %h_i, %m_i, %g, %d)
        %flux_r:2 = call @compute_flux(%h_i, %m_i, %h_ip1, %m_ip1, %g, %d)
        
        %dh = arith.divf (arith.subf %flux_l#0, %flux_r#0), %dx : f32
        %dm = arith.divf (arith.subf %flux_l#1, %flux_r#1), %dx : f32
        
        memref.store %dh, %k1_h[%i] : memref<?xf32>
        memref.store %dm, %k1_m[%i] : memref<?xf32>
    }

    
    // 准备计算k2的中间值: temp = u + dt*k1/2
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %h[%i] : memref<?xf32>
        %m_i = memref.load %m[%i] : memref<?xf32>
        %k1h = memref.load %k1_h[%i] : memref<?xf32>
        %k1m = memref.load %k1_m[%i] : memref<?xf32>
        
        %dt_half = arith.mulf %dt, %c0_5 : f32
        %dh = arith.mulf %k1h, %dt_half : f32
        %dm = arith.mulf %k1m, %dt_half : f32
        %new_h = arith.addf %h_i, %dh : f32
        %new_m = arith.addf %m_i, %dm : f32
        
        memref.store %new_h, %temp_h[%i] : memref<?xf32>
        memref.store %new_m, %temp_m[%i] : memref<?xf32>
    }

    // 计算k2 (使用temp数组)
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %temp_h[%i] : memref<?xf32>
        %m_i = memref.load %temp_m[%i] : memref<?xf32>
        %h_ip1 = memref.load %temp_h[%i+1] : memref<?xf32>
        %m_ip1 = memref.load %temp_m[%i+1] : memref<?xf32>
        %h_im1 = memref.load %temp_h[%i-1] : memref<?xf32>
        %m_im1 = memref.load %temp_m[%i-1] : memref<?xf32>
        
        %flux_l:2 = call @compute_flux(%h_im1, %m_im1, %h_i, %m_i, %g, %d)
        %flux_r:2 = call @compute_flux(%h_i, %m_i, %h_ip1, %m_ip1, %g, %d)
        
        %dh = arith.divf (arith.subf %flux_l#0, %flux_r#0), %dx : f32
        %dm = arith.divf (arith.subf %flux_l#1, %flux_r#1), %dx : f32
        
        memref.store %dh, %k2_h[%i] : memref<?xf32>
        memref.store %dm, %k2_m[%i] : memref<?xf32>
    }

    // 准备计算k3的中间值: temp = u + dt*k2/2
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %h[%i] : memref<?xf32>
        %m_i = memref.load %m[%i] : memref<?xf32>
        %k2h = memref.load %k2_h[%i] : memref<?xf32>
        %k2m = memref.load %k2_m[%i] : memref<?xf32>
        
        %dt_half = arith.mulf %dt, %c0_5 : f32
        %dh = arith.mulf %k2h, %dt_half : f32
        %dm = arith.mulf %k2m, %dt_half : f32
        %new_h = arith.addf %h_i, %dh : f32
        %new_m = arith.addf %m_i, %dm : f32
        
        memref.store %new_h, %temp_h[%i] : memref<?xf32>
        memref.store %new_m, %temp_m[%i] : memref<?xf32>
    }

    // 计算k3 (使用temp数组)
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %temp_h[%i] : memref<?xf32>
        %m_i = memref.load %temp_m[%i] : memref<?xf32>
        %h_ip1 = memref.load %temp_h[%i+1] : memref<?xf32>
        %m_ip1 = memref.load %temp_m[%i+1] : memref<?xf32>
        %h_im1 = memref.load %temp_h[%i-1] : memref<?xf32>
        %m_im1 = memref.load %temp_m[%i-1] : memref<?xf32>
        
        %flux_l:2 = call @compute_flux(%h_im1, %m_im1, %h_i, %m_i, %g, %d)
        %flux_r:2 = call @compute_flux(%h_i, %m_i, %h_ip1, %m_ip1, %g, %d)
        
        %dh = arith.divf (arith.subf %flux_l#0, %flux_r#0), %dx : f32
        %dm = arith.divf (arith.subf %flux_l#1, %flux_r#1), %dx : f32
        
        memref.store %dh, %k3_h[%i] : memref<?xf32>
        memref.store %dm, %k3_m[%i] : memref<?xf32>
    }

    // 准备计算k4的中间值: temp = u + dt*k3
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %h[%i] : memref<?xf32>
        %m_i = memref.load %m[%i] : memref<?xf32>
        %k3h = memref.load %k3_h[%i] : memref<?xf32>
        %k3m = memref.load %k3_m[%i] : memref<?xf32>
        
        %dh = arith.mulf %k3h, %dt : f32
        %dm = arith.mulf %k3m, %dt : f32
        %new_h = arith.addf %h_i, %dh : f32
        %new_m = arith.addf %m_i, %dm : f32
        
        memref.store %new_h, %temp_h[%i] : memref<?xf32>
        memref.store %new_m, %temp_m[%i] : memref<?xf32>
    }

    // 计算k4 (使用temp数组)
    scf.for %i = %c0 to %div step %c1 {
        %h_i = memref.load %temp_h[%i] : memref<?xf32>
        %m_i = memref.load %temp_m[%i] : memref<?xf32>
        %h_ip1 = memref.load %temp_h[%i+1] : memref<?xf32>
        %m_ip1 = memref.load %temp_m[%i+1] : memref<?xf32>
        %h_im1 = memref.load %temp_h[%i-1] : memref<?xf32>
        %m_im1 = memref.load %temp_m[%i-1] : memref<?xf32>
        
        %flux_l:2 = call @compute_flux(%h_im1, %m_im1, %h_i, %m_i, %g, %d)
        %flux_r:2 = call @compute_flux(%h_i, %m_i, %h_ip1, %m_ip1, %g, %d)
        
        %dh = arith.divf (arith.subf %flux_l#0, %flux_r#0), %dx : f32
        %dm = arith.divf (arith.subf %flux_l#1, %flux_r#1), %dx : f32
        
        memref.store %dh, %k4_h[%i] : memref<?xf32>
        memref.store %dm, %k4_m[%i] : memref<?xf32>
    }

    // 释放临时数组
    memref.dealloc %temp_h
    memref.dealloc %temp_m
    
    // RK4最终更新
    scf.for %i = %c0 to %div step %c1 {
    %h_old = memref.load %h[%i] : memref<?xf32>
    %m_old = memref.load %m[%i] : memref<?xf32>
    
    %k1h = memref.load %k1_h[%i] : memref<?xf32>
    %k2h = memref.load %k2_h[%i] : memref<?xf32>
    %k3h = memref.load %k3_h[%i] : memref<?xf32>
    %k4h = memref.load %k4_h[%i] : memref<?xf32>
    
    %k1m = memref.load %k1_m[%i] : memref<?xf32>
    %k2m = memref.load %k2_m[%i] : memref<?xf32>
    %k3m = memref.load %k3_m[%i] : memref<?xf32>
    %k4m = memref.load %k4_m[%i] : memref<?xf32>

    // RK4更新公式 for h
    %h_new_val = arith.addf %h_old, 
        arith.mulf %dt, 
        arith.divf (arith.addf %k1h, 
                arith.addf (arith.mulf %c2, %k2h),
                arith.addf (arith.mulf %c2, %k3h),
                %k4h), %c6 : f32
                
    // RK4更新公式 for m
    %m_new_val = arith.addf %m_old,
        arith.mulf %dt,
        arith.divf (arith.addf %k1m,
                arith.addf (arith.mulf %c2, %k2m),
                arith.addf (arith.mulf %c2, %k3m),
                %k4m), %c6 : f32
    
    memref.store %h_new_val, %h_new[%i] : memref<?xf32>
    memref.store %m_new_val, %m_new[%i] : memref<?xf32>
    }
    
    memref.dealloc %k1_h, %k1_m // ... 释放临时数组
    return
  }
}
